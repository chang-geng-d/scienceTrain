import numpy as np
import keras
import time
import pickle
import codecs
from keras import Input, Model, Sequential
# from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from BFV import encrypt
from BFV import bfv
import key_generation
from SM9.gmssl import sm9


class LocalModel(object):
    def __init__(self, model_config):
        self.training_start_time = int(round(time.time()))
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        # self.model.summary()
        self.discriminator = self.model
        # self.discriminator.summary()
        # optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        # TODO:改为传统mnist
        self.latent_dim = 100
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.discriminator.compile(optimizer=keras.optimizers.Adam(lr=0.005),
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])

    def train(self, batch_size, req_id):
        # TODO:改变训练集后需要检查
        # (X_train, y_train), (_, _) = mnist.load_data()
        pne_data = np.load('./DataSet/bloodmnist.npz')
        print(pne_data.files)
        x_train = pne_data['train_images']
        y_train = pne_data['train_labels']
        x_test = pne_data['test_images']
        y_test = pne_data['test_labels']
        x_val = pne_data['val_images']
        y_val = pne_data['val_labels']
        print('Train: ', x_train.shape, y_train.shape)
        print('Test: ', x_test.shape, y_test.shape)
        print('Val: ', x_val.shape, y_val.shape)
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_val = x_val / 255.0
        if req_id == 1:
            x_train = x_train[:6000]
            y_train = y_train[:6000]
        else:
            x_train = x_train[6000:]
            y_train = y_train[6000:]
        # for j in range(8):
        #     for i in range(len(y_val)):
        #         if y_val[i] == j:
        #             plt.imshow(x_val[i])
        #             plt.show()
        #             break
        # w1 = self.discriminator.get_weights()[0]
        # print(w1.shape)
        self.discriminator.fit(x_train, y_train, epochs=1, batch_size=64, shuffle=True)
        test = self.discriminator.evaluate(x_test, y_test, verbose=0)
        f = open("result.txt", "a")
        print(f"test_loss:{test[0]}, test_acc:{test[1]}", file=f)
        f.close()

        return self.discriminator.get_weights(), test[0], test[1]

    def get_weights(self):
        return self.model.get_weights()

    def set_weights2(self, new_weights):
        self.discriminator.set_weights(new_weights)


class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200  # 训练的最大数据集大小？

    def __init__(self, server_host, server_port):
        # TODO:存在冗余信息,可改
        self.local_model = None
        self.req_num = 1
        self.id = None
        self.num = 0
        self.t1 = 0
        self.t2 = 0

        # 生成公私钥对
        # self.pk, self.sk = BFV.generate_keypair(128)
        self.pk = bfv.publicKey(key_generation.N, key_generation.G)
        self.sk = bfv.privateKey(key_generation.N, key_generation.LAM, key_generation.G)
        print("public_key.n:", self.pk.n)
        print("public_key.g:", self.pk.g)
        print("secret_key.lam:", self.sk.lam)

        # sm9
        self.master_public, self.master_secret = sm9.setup('sign') #生成客户端的公私钥

        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print("启动客户端(sent wakeup)")
        self.t1 = time.time()
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def on_init(self, *args):
        '''
        接收服务器发送的模型,初始化本地模型,并请求更新
        '''
        self.t2 = time.time()
        f = open("expense.txt", "a")
        print(f"认证总用时:{self.t2 - self.t1}s", file=f)
        f.close()
        model_config = args[0]
        print('初始化(on init)', model_config)
        self.local_model = LocalModel(model_config)

        # ready to be dispatched for training
        print('请求更新(preparing sent client_ready)')
        self.sio.emit('client_ready', {
            'class_distr': 1
        })

    def on_authentication(self, *args):
        '''
        进行签名认证,并回发签名与公钥
        '''
        req = args[0]
        self.id = req['id']
        self.num = req['num']
        print('开始签名...')
        t1 = time.time()
        # sm9签名认证
        # if self.num == 1:
        # master_public, master_secret = key_generation.P_A, key_generation.S_A
        print(f"master_public:{self.master_public}")
        print(f"master_secret:{self.master_secret}")
        ida = f"{self.id}"
        print(f"id:{ida}")
        Da = sm9.private_key_extract('sign', self.master_public, self.master_secret, ida)
        print(f"DA:{Da}")
        message = 'connect'
        print(f"message:{message}")
        signature = sm9.sign(self.master_public, Da, message)
        print(f"signature:{signature}")
        # elif self.num == 2:
        #     # master_public, master_secret = key_generation.P_B, key_generation.S_B
        #     print(f"master_public:{master_public}")
        #     print(f"master_secret:{master_secret}")
        #     idb = f"{self.id}"
        #     Da = sm9.private_key_extract('sign', master_public, master_secret, idb)
        #     print(f"DA:{Da}")
        #     message = 'connect'
        #     print(f"message:{message}")
        #     signature = sm9.sign(master_public, Da, message)
        #     print(f"signature:{signature}")
        print("签名完成...")
        t2 = time.time()
        f = open("expense.txt", "a")
        print(f"用户签名计算用时:{t2 - t1}s", file=f)
        f.close()
        self.sio.emit('signature', {
            'signature': obj_to_pickle_string(signature),
            'master_public': obj_to_pickle_string(self.master_public),
        })

    def register_handles(self):
        '''
        注册客户机处理句柄
        '''
        def req_train(*args):
            '''
            接收服务器ACK,重新回到ready状态等待服务器下一次更新请求
            '''
            # time.sleep(2)
            # print('当前时间：',time.strftime('%H-%M-%S',time.localtime(time.time())))
            self.req_num += 1
            print('请求第' + str(self.req_num) + '次更新：')
            self.sio.emit('client_ready', {'req_num': self.req_num})

        def on_connect():   # debug用
            print('connect')

        def on_disconnect():    # debug用
            print('disconnect')

        def on_reconnect(): # debug用
            print('reconnect')

        def on_success(*args):
            '''
            接收服务器的成功消息,并回发模型拉取请求
            '''
            print("签名认证成功...")
            self.sio.emit('client_connect')

        def on_request_update(*args):
            '''
            接收服务器传参,并对本地的模型进行一轮训练,回发本地模型权值
            '''
            req = args[0]
            round_number1 = str(req['round_number'])
            print('总轮数：' + round_number1)

            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            # 初始训练参数不需要解密
            if req['round_number'] != 1:
                print("解密中...")
                t3 = time.time()
                weights = encrypt.decryption(self.sk, weights)
                t4 = time.time()
                f = open("expense.txt", "a")
                print(f"解密用时:{t4 - t3}s", file=f)
                f.close()
                print("解密完成,开始训练...")
                print(f"weights:{weights}")

            self.local_model.set_weights2(weights)

            my_weights, train_loss, train_accuracy = self.local_model.train(batch_size=128, req_id=self.id)

            # f1 = open("weights.txt", "w")
            # print(f"weights:{my_weights}", file=f1)

            print(f"weights:{my_weights}")
            # 给参数加密
            t1 = time.time()
            print("正在加密...")
            my_weights = encrypt.encryption(self.pk, my_weights)

            # f2 = open("en_weights.txt", "w")
            # print(f"en_weights:{my_weights}", file=f2)

            t2 = time.time()
            f = open("expense.txt", "a")
            print(f"加密用时:{t2 - t1}s", file=f)
            f.close()
            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
            }
            print("加密完成，上传参数中...")
            self.sio.emit('client_update', resp)
        # TODO:绑定风格不同,可改
        self.sio.on('next', req_train)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('authentication', lambda *args: self.on_authentication(*args))
        self.sio.on('success', on_success)


def obj_to_pickle_string(x):
    '''
    python对象转为2进制并序列化为base64
    '''
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    '''
    从base64反序列化python对象
    '''
    return pickle.loads(codecs.decode(s.encode(), "base64"))  # 模型返序列化loads，编解码en/decode


if __name__ == "__main__":
    # TODO:准备改为嵌入式
    client = FederatedClient("127.0.0.1", 5000)
