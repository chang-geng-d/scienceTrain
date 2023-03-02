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


class GlobalModel(object):
    """docstring for GlobalModel"""

    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        self.img_shape = (28, 28, 3)

    def build_model(self):
        # raise NotImplementedError()
        pass

    def update_weights(self, client_weights):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        for i in range(len(new_weights)):
            new_weights[i] = client_weights[i]
        self.current_weights = new_weights
        print('服务器更新成功！')


class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
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

        self.ready_client_sids = set()
        self.flag = 1
        self.first_id = 0
        self.second_id = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())  # 随机生成uuid
        self.num = 0
        self.current_round = 0
        self.current_round_client_updates = []
        self.client_updates_weights = []
        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):

        @self.socketio.on('connect')
        def handle_connect():
            pass
            # print(request.sid[0], "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            self.num += 1
            print("开始身份认证：", request.sid)
            emit('authentication', {
                'id': request.sid,
                'num': self.num,
            })

        @self.socketio.on('signature')
        def handle_signature(data):
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
            print("认证成功,开始连接客户: ", request.sid)
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            # print("client ready for training", request.sid, data)
            print('客户端', request.sid, '请求更新')
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) == 2:
                self.first_id = request.sid
                for idd in self.ready_client_sids:
                    if idd != self.first_id:
                        self.secend_id = idd
                self.train_next_round(self.first_id)

        @self.socketio.on('client_update')
        def handle_client_update(data):
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
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))  # 模型返序列化loads，编解码en/decode


if __name__ == '__main__':
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000")
    server.start()
