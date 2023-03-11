# -*- coding:utf-8 -*-
import pickle
import keras
import uuid
from keras.models import Sequential
import codecs
import json
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D
from flask import *
from flask_socketio import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# turn off url logging
import logging
from SM9.gmssl import sm9


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class GlobalModel(object):  #虚父类
    """docstring for GlobalModel"""

    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()

    def build_model(self):
        pass

    def update_weights(self, client_weights):
        '''
        更新权值(未加载),由于不知道客户端数目,故不在此做平均
        '''
        for i in range(len(client_weights)):
            self.current_weights[i]+=client_weights[i]
    
    def update_weights2(self,clientNum):
        for i in range(len(self.current_weights)):
            self.current_weights[i]=self.current_weights[i]/clientNum
        self.model.set_weights(self.current_weights)


class GlobalModel_MNIST_CNN(GlobalModel):
    '''
    此处使用的数据集为blood mnist(28*28*3大小,8类别)
    '''
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        '''
        28*28*1输入,10概率输出
        '''
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        # model.summary()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        return model


class FLServer(object):
    def __init__(self, global_model, host, port):
        self.global_model=global_model()

        self.ready_client_sids=set()    # 已连接并就绪的客户端sid
        self.block_client_sids=set()    # 等待回发更新的客户端sid
        self.block_client_num=0

        self.app=Flask(__name__)
        self.socketio=SocketIO(self.app)
        self.host=host
        self.port=port
        self.model_id=str(uuid.uuid4()) # 随机生成uuid
        self.current_round=0            # 全局模型训练轮数
        self.regist_connect_handles()
        self.regist_update_handles()

        print(f'当前全局模型uuid: {self.model_id}')

        @self.app.route('/')
        def dashboard():    # 无用
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def regist_connect_handles(self):

        @self.socketio.on('connect')
        def handle_connect():   # debug用
            '''
            客户端连接
            '''
            print(f"用户 {request.sid} 连接")

        @self.socketio.on('reconnect')
        def handle_reconnect(): # debug用
            '''
            客户端重连
            '''
            print(f"用户 {request.sid} 重连")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            '''
            客户端断连,将客户端对应的sid从连接集合中去除
            '''
            print(f"用户 {request.sid} 断开连接")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            '''
            客户端启动,回送认证请求
            '''
            print(f"开始用户 {request.sid} 身份认证")
            emit('authentication', {
                'id': request.sid
            })

        @self.socketio.on('signature')
        def handle_signature(data):
            '''
            对客户机回发的签名和公钥进行验证,并回发成功消息
            '''
            print(f"开始验证用户 {request.sid} 签名...")
            master_public = self.pickle_string_to_obj(data['master_public'])
            idA = f"{request.sid}"
            message = "connect"
            signature = self.pickle_string_to_obj(data['signature'])
            # print(f"master_public:{master_public}")
            # print(f"message:{message}")
            # print(f"id:{idA}")
            # print(f"signature:{signature}")
            assert (sm9.verify(master_public, idA, message, signature)) #断言验证
            print("认证成功...")
            emit('success')

        @self.socketio.on('client_ready')
        def handle_client_ready():
            '''
            接收客户机请求,回发json格式模型与当前模型id
            '''
            print(f'开始初始化用户 {request.sid} 模型')
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
            },callback=after_connect)

        def after_connect():
            print(f'用户 {request.sid} 本地模型初始化成功')
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids)==2:  # debug用
                self.train_next_round()

    def regist_update_handles(self):

        @self.socketio.on('client_update')
        def handle_client_update(data):
            '''
            接收客户端权值,更新全局模型,并回发ACK
            '''
            print("处理客户", request.sid,'的更新')
            if self.current_round==data['round_number']:
                client_updates_weights = self.pickle_string_to_obj(data['weights'])
                self.global_model.update_weights(client_updates_weights)
                emit('update_ack')
                self.ready_client_sids.add(request.sid)
                self.block_client_sids.discard(request.sid)
                if not self.block_client_sids:
                    self.global_model.update_weights2(self.block_client_num)
                    print(f'所有 {self.block_client_num} 个客户机更新完毕')

    def train_next_round(self):
        '''
        服务器轮次计数器加1,并请求客户端的更新权值以便聚合
        '''
        self.current_round+=1
        print("全局模型轮次: ", self.current_round)
        emit('request_update',{
            'model_id': self.model_id,
            'round_number': self.current_round,
            'current_weights': self.obj_to_pickle_string(self.global_model.current_weights)
        },broadcast=True)
        self.block_client_sids=self.ready_client_sids.copy()
        self.block_client_num=len(self.block_client_sids)
        self.ready_client_sids.clear()

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)

    def obj_to_pickle_string(self,x):
        '''
        python对象转为2进制并序列化为base64
        '''
        return codecs.encode(pickle.dumps(x), "base64").decode()

    def pickle_string_to_obj(self,s):
        '''
        从base64反序列化python对象
        '''
        return pickle.loads(codecs.decode(s.encode(), "base64"))


if __name__ == '__main__':
    server=FLServer(GlobalModel_MNIST_CNN,'127.0.0.1',5000)
    server.start()