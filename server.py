# -*- coding:utf-8 -*-
import pickle
import time

import keras
import uuid
from keras.models import Sequential
import codecs
import numpy as np
import json
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D
from flask import *
from flask_socketio import *

# turn off url logging
import logging

from BFV import bfv
from SM9.gmssl import sm9
import key_generation

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class GlobalModel(object):  #虚父类
    """docstring for GlobalModel"""

    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        # TODO:无用
        self.img_shape = (28, 28, 3)    # blood mnist单输入大小

    def build_model(self):
        # raise NotImplementedError()
        pass

    def update_weights(self, client_weights):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        # FIXME:此处权值更新操作有误
        for i in range(len(new_weights)):
            new_weights[i] = client_weights[i]
        self.current_weights = new_weights
        # FIXME:此处权值更新完毕后未加载至当前模型
        print('服务器更新成功！')


class GlobalModel_MNIST_CNN(GlobalModel):
    '''
    此处使用的数据集为blood mnist(28*28*3大小,8类别)
    '''
    # TODO:改为传统mnist
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        '''
        28*28*3输入,8概率输出
        '''
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(lr=0.005),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


class FLServer(object):
    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()  # 已连接的客户端sid集合

        # TODO:演示所用,将客户端更新范围限制于first_id和second_id所指定的2个客户端,flag用于管理更新切换,可改
        self.flag = 1
        self.first_id = 0   
        self.second_id = 0

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())  # 随机生成uuid
        self.num = 0    # 已连接的客户端数量
        self.current_round = 0  # 全局模型训练轮数
        self.current_round_client_updates = []  # 当前轮数的到的客户端更新数据,包含轮数及base64编码的权重信息
        self.client_updates_weights = []    # 当前轮数得到的解码后的更新权重
        self.register_handles()

        @self.app.route('/')
        def dashboard():    # 无用
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):

        @self.socketio.on('connect')
        def handle_connect():   # debug用
            '''
            客户端连接
            '''
            pass
            # print(request.sid[0], "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect(): # debug用
            '''
            客户端重连
            '''
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            '''
            客户端断连,将客户端对应的sid从连接集合中去除
            '''
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            '''
            客户端启动,回送认证请求
            '''
            self.num += 1
            print("开始身份认证：", request.sid)
            emit('authentication', {
                'id': request.sid,
                'num': self.num,
            })

        @self.socketio.on('signature')
        def handle_signature(data):
            '''
            对客户机回发的签名和公钥进行验证,并回发成功消息
            '''
            print("开始验证签名...")
            t1 = time.time()
            # if self.num == 1:
            # master_public = key_generation.P_A
            # req = data[0]
            master_public = pickle_string_to_obj(data['master_public'])
            idA = f"{request.sid}"
            message = "connect"
            signature = pickle_string_to_obj(data['signature'])
            print(f"master_public:{master_public}")
            print(f"message:{message}")
            print(f"id:{idA}")
            print(f"signature:{signature}")
            assert (sm9.verify(master_public, idA, message, signature))
            print("认证成功...")
            t2 = time.time()
            f = open("expense.txt", "a")
            print(f"服务器验证签名用时:{t2 - t1}s", file=f)
            f.close()
            # elif self.num == 2:
            #     master_public = key_generation.P_B
            #     idB = f"{request.sid}"
            #     message = "connect"
            #     signature = data
            #     assert (sm9.verify(master_public, idB, message, signature))
            emit('success')

        @self.socketio.on('client_connect')
        def handle_wake_up():
            '''
            接收客户机请求,回发json格式模型与当前模型id
            '''
            print("认证成功,开始连接客户: ", request.sid)
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            '''
            将已完成认证并初始化的客户机加入已连接集合中,此时新建连接过程完成
            '''
            # print("client ready for training", request.sid, data)
            print('客户端', request.sid, '请求更新')
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) == 2:    #当正好有两个客户机连接时，初始化first_id和second_id,并进行一轮训练
                self.first_id = request.sid
                for idd in self.ready_client_sids:
                    if idd != self.first_id:
                        self.secend_id = idd
                self.train_next_round(self.first_id)

        @self.socketio.on('client_update')
        def handle_client_update(data):
            '''
            接收客户端权值,更新全局模型,并回发ACK
            '''
            print("处理客户", request.sid, '的更新')
            self.current_round_client_updates = [data]
            self.client_updates_weights = pickle_string_to_obj(data['weights'])
            print(self.client_updates_weights)
            self.global_model.update_weights(self.client_updates_weights)
            if self.flag == 1:
                self.flag == 2
                emit('next', room=self.secend_id)
            else:
                self.flag==1
                emit('next', room=self.first_id)

    def train_next_round(self, id):
        '''
        服务器轮次计数器加1,并请求客户端的更新权值以便聚合
        '''
        self.current_round += 1
        self.current_round_client_updates = []
        print("总轮次 ", self.current_round)
        emit('request_update', {
            'model_id': self.model_id,
            'round_number': self.current_round,
            'current_weights': obj_to_pickle_string(self.global_model.current_weights),
            'weights_format': 'pickle',
        }, room=id)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x):
    '''
    python对象转为2进制并序列化为base64
    '''
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    '''
    从base64反序列化python对象
    '''
    return pickle.loads(codecs.decode(s.encode(), "base64"))


if __name__ == '__main__':
    # TODO:准备改为嵌入式
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000")
    server.start()
