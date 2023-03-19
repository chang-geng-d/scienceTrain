# -*- coding:utf-8 -*-
import os
import uuid
import keras
import codecs
import pickle
import threading
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D
from flask import *
from flask_socketio import *
from SM9.gmssl import sm9
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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


class FLServer(threading.Thread):
    def __init__(self,host, port):
        threading.Thread.__init__(self)
        self.global_model=GlobalModel_MNIST_CNN()

        self.ready_client_sids=set()    # 已连接并就绪的客户端sid
        self.block_client_sids=set()    # 等待回发更新的客户端sid
        self.block_client_num=0

        self.isRunning=False
        self.app=Flask(__name__)
        self.sio=SocketIO(self.app)
        self.host=host
        self.port=port
        self.model_id=str(uuid.uuid4()) # 随机生成uuid
        self.current_round=0            # 全局模型训练轮数
        self.regist_connect_handles()
        self.regist_update_handles()

        self.serverLog=Logger('logs/fl_server','server.log')
        self.userLogs={}

        self.serverLog.log.info(f'当前全局模型uuid: {self.model_id}')

        self.setDaemon(True)

    def regist_connect_handles(self):
        
        @self.sio.on('connect')
        def handle_connect():
            '''
            客户端连接
            '''
            log=Logger('logs/fl_server',f'{request.sid}.log')
            self.userLogs[request.sid]=log
            log.log.info(f"用户 {request.sid} 连接")

        @self.sio.on('reconnect')
        def handle_reconnect(): # debug用
            '''
            客户端重连
            '''
            self.userLogs[request.sid].log.info(f"用户 {request.sid} 重连")

        @self.sio.on('disconnect')
        def handle_reconnect():
            '''
            客户端断连,将客户端对应的sid从连接集合中去除
            '''
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)
                self.userLogs[request.sid].log.info(f"用户 {request.sid} 断开连接")

        @self.sio.on('client_wake_up')
        def handle_wake_up():
            '''
            客户端启动,回送认证请求
            '''
            self.userLogs[request.sid].log.info(f"开始用户 {request.sid} 身份认证")
            emit('authentication', {
                'id': request.sid
            })

        @self.sio.on('signature')
        def handle_signature(data):
            '''
            对客户机回发的签名和公钥进行验证,并回发成功消息
            '''
            self.userLogs[request.sid].log.info(f"开始验证用户 {request.sid} 签名...")
            master_public = self.pickle_string_to_obj(data['master_public'])
            idA = f"{request.sid}"
            message = "connect"
            signature = self.pickle_string_to_obj(data['signature'])
            # print(f"master_public:{master_public}")
            # print(f"message:{message}")
            # print(f"id:{idA}")
            # print(f"signature:{signature}")
            assert (sm9.verify(master_public, idA, message, signature)) #断言验证
            self.userLogs[request.sid].log.info("认证成功...")
            emit('success')

        @self.sio.on('client_ready')
        def handle_client_ready():
            '''
            接收客户机请求,回发json格式模型与当前模型id
            '''
            self.userLogs[request.sid].log.info(f'开始初始化用户 {request.sid} 模型')
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
            },callback=after_connect)

        def after_connect():
            self.userLogs[request.sid].log.info(f'用户 {request.sid} 本地模型初始化成功')
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids)==2:  # debug用
                self.train_next_round()

    def regist_update_handles(self):

        @self.sio.on('client_update')
        def handle_client_update(data):
            '''
            接收客户端权值,更新全局模型,并回发ACK
            '''
            self.userLogs[request.sid].log.info(f"处理客户 {request.sid} 的更新")
            if self.current_round==data['round_number']:
                client_updates_weights = self.pickle_string_to_obj(data['weights'])
                self.global_model.update_weights(client_updates_weights)
                emit('update_ack')
                self.ready_client_sids.add(request.sid)
                self.block_client_sids.discard(request.sid)
                if not self.block_client_sids:
                    self.global_model.update_weights2(self.block_client_num)
                    self.serverLog.log.info(f'所有 {self.block_client_num} 个客户机更新完毕')

    def train_next_round(self):
        '''
        服务器轮次计数器加1,并请求客户端的更新权值以便聚合
        '''
        if self.block_client_sids:
            return
        self.current_round+=1
        self.serverLog.log.info(f"全局模型轮次: {self.current_round}")
        emit('request_update',{
            'model_id': self.model_id,
            'round_number': self.current_round,
            'current_weights': self.obj_to_pickle_string(self.global_model.current_weights)
        },broadcast=True)
        self.block_client_sids=self.ready_client_sids.copy()
        self.block_client_num=len(self.block_client_sids)
        self.ready_client_sids.clear()

    def run(self):
        if not self.isRunning:
            print('enter run')
            self.isRunning=True
            self.sio.run(self.app,host=self.host,port=self.port)
    
    def stop(self):
        if self.isRunning:
            self.ready_client_sids.clear()
            # self.sio.stop()   #关不掉,只能这样了,将其设置为守护，至少能在主线程退出时不会残留
            self.isRunning=False
    
    def delLog(self,stop=True):
        if not self.isRunning:
            self.userLogs.clear()
            if not stop:
                del self.serverLog

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

# if __name__ == '__main__':
#     server=FLServer('127.0.0.1',9000)
#     server.start()
#     print('aaa')