import os
import keras
import pickle
import codecs
import numpy as np
import tensorflow as tf
from keras import losses
from keras import optimizers
from keras.datasets import mnist
from keras.models import model_from_json
import socketio
from BFV import bfv
import key_generation
from SM9.gmssl import sm9
from logger import Logger

from gan_attack.gan import GAN
from membership_attack.mAtk_model import MATK_MODEL
from PPA_attack.pAtk_model import PATK_MODEL

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class LocalModel(object):
    def __init__(self,model_id,model_config):
        self.model_id=model_id
        self.model=model_from_json(model_config)
        self.loss=losses.binary_crossentropy
        self.model.compile(optimizer=optimizers.Adam(0.0002, 0.5),loss=self.loss,metrics=['accuracy'])

    def train(self):
        '''
        训练五次
        '''
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        
        train_pos=np.random.randint(0,x_train.shape[0],1024) #在保证运行效率同时保证攻击效果
        x_train = x_train[train_pos] /127.5-1
        x_test = x_test /127.5-1
        y_train=tf.one_hot(indices=y_train[train_pos],depth=10,axis=1).numpy()
        y_test=tf.one_hot(indices=y_test,depth=10,axis=1).numpy()

        self.model.fit(x_train, y_train, epochs=5, batch_size=64, shuffle=True)
        # test = self.model.evaluate(x_test, y_test, verbose=0)
        return self.model.get_weights()#, test[0], test[1]

    def train_laplaceNoise(self):
        (trainX,trainY),(_,_)=mnist.load_data()
        trainPos=np.random.randint(0,trainX.shape[0],1024)
        trainX=trainX[trainPos]/127.5-1
        trainY=tf.one_hot(indices=trainY[trainPos],depth=10,axis=1).numpy()

        batchSize=64
        stepsPerEpoch=trainX.shape[0]//batchSize

        for epoch in range(5):
            for step in range(stepsPerEpoch):
                idx=np.random.randint(0,trainX.shape[0],batchSize)
                with tf.GradientTape() as tape:
                    predY=self.model(trainX[idx,:,:],training=True)
                    loss=self.loss(trainY[idx,:],predY)
                trainAbleVars=self.model.trainable_variables
                grads=tape.gradient(loss,trainAbleVars) #在tf2.10.0中当loss极小时求导会在卷积核处大小失真报错
                grads=[tf.clip_by_value(tf.add(x,tf.random.normal(shape=x.shape,mean=0.0,stddev=0.7,dtype=tf.float32)),-1,1) for x in grads]
                self.model.optimizer.apply_gradients(zip(grads,trainAbleVars))
                self.metric.update_state(trainY[idx,:],predY)
        
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self,new_weights):
        self.model.set_weights(new_weights)

class FederatedClient(object):
    def __init__(self, server_host, server_port):
        # 训练所用本地模型及轮数
        self.local_model:LocalModel=None
        self.round_num=0

        # 生成公私钥对,用于模型权重加密
        self.pk = bfv.publicKey(key_generation.N, key_generation.G)
        self.sk = bfv.privateKey(key_generation.N, key_generation.LAM, key_generation.G)

        # sm9,用于连接认证
        self.master_public, self.master_secret = sm9.setup('sign')

        # 客户端参数
        self.isConnected=False
        self.sio=socketio.Client()
        self.sHost=server_host
        self.sPort=server_port
        self.sid=None
        self.regist_connect_handles()
        self.regist_update_handles()

        self.userLog=None

        self.isBad=False
        self.method='none'
        # 攻击
        self.ganAttackModel=None
        self.membershipModel=None
        self.ppaAtkModel=None

        # 防御
        # self.pubKey,self.priKey=paillier.generate_paillier_keypair()

    def regist_connect_handles(self):
        '''
        注册客户机处理句柄
        '''
        @self.sio.on('connect')
        def on_connect():   # debug用
            # print('connect')
            pass

        @self.sio.on('disconnect')
        def on_disconnect():    # debug用
            # print('disconnect')
            pass

        @self.sio.on('reconnect')
        def on_reconnect(): # debug用
            # print('reconnect')
            pass

        @self.sio.on('authentication')
        def on_authentication(data):
            '''
            进行签名认证,并回发签名与公钥
            '''
            self.sid=data['id']
            self.userLog=Logger('logs/fl_client',f'{self.sid}.log')
            self.userLog.log.info('连接完成,开始签名...')
            # sm9签名认证
            # print(f"master_public:{self.master_public}")
            # print(f"master_secret:{self.master_secret}")
            ida = f"{self.sid}"
            # print(f"id:{ida}")
            Da = sm9.private_key_extract('sign', self.master_public, self.master_secret, ida)
            # print(f"DA:{Da}")
            message = 'connect'
            # print(f"message:{message}")
            signature = sm9.sign(self.master_public, Da, message)
            # print(f"signature:{signature}")
            self.userLog.log.info("签名完成,发送签名进行认证...")
            self.sio.emit('signature', {
                'signature': self.obj_to_pickle_string(signature),
                'master_public': self.obj_to_pickle_string(self.master_public),
            })

        @self.sio.on('success')
        def on_success():
            '''
            接收服务器的成功消息,并回发模型拉取请求
            '''
            self.userLog.log.info("服务器端签名认证成功")
            self.sio.emit('client_ready')

        @self.sio.on('init')
        def on_init(data):
            '''
            接收服务器发送的模型,初始化本地模型,并请求更新
            '''
            self.userLog.log.info(f"初始化模型:{data['model_id']}")
            self.local_model = LocalModel(data['model_id'],data['model_json'])

    def regist_update_handles(self):

        @self.sio.on('request_update')
        def on_request_update(data):
            '''
            接收服务器传参,并对本地的模型进行一轮训练,回发本地模型权值
            '''
            self.round_num=data['round_number']
            weights=self.pickle_string_to_obj(data['current_weights'])
            cNum=data['client_num'] 

            # 初始训练参数不需要解密
            # if data['round_number'] != 1:
            #     print("解密中...")
            #     weights = encrypt.decryption(self.sk, weights)
            #     print("解密完成,开始训练...")
            #     print(f"weights:{weights}")

            self.local_model.set_weights(weights)
            if not self.isBad:
                if self.method=='none':
                    self.userLog.log.info(f'开始训练,当前模型id为: {self.local_model.model_id}')
                    my_weights = self.local_model.train()
                elif self.method=='diffPri':
                    self.userLog.log.info(f'开始训练,当前模型id为: {self.local_model.model_id}')
                    my_weights=self.local_model.train_laplaceNoise()
            else:
                if self.method=='gan':
                    self.userLog.log.info(f'开始GAN攻击,当前模型id为: {self.local_model.model_id}')
                    self.gan_attack(self.round_num)
                    my_weights=self.local_model.get_weights()
                elif self.method=='membership':
                    self.userLog.log.info(f'开始更新成员推断攻击模型,当前模型id为:{self.local_model.model_id}')
                    self.membership_attack(weights)
                elif self.method=='PPA':
                    self.userLog.log.info(f'开始更新PPA攻击模型,当前模型id为:{self.local_model.model_id}')
                    self.ppa_attack(weights)

            for i in range(len(my_weights)):
                my_weights[i]=np.array([x/cNum for x in my_weights[i]])

            # print(f"weights:{my_weights}")
            # # 给参数加密
            # print("正在加密...")
            # my_weights = encrypt.encryption(self.pk, my_weights)
            # print("加密完成，上传参数中...")
            self.sio.emit('client_update',{
                'round_number': self.round_num,
                'weights': self.obj_to_pickle_string(my_weights),
            })
            self.userLog.log.info("上传参数完成")

        @self.sio.on('update_ack')
        def req_train():
            '''
            接收服务器ACK,重新回到ready状态等待服务器下一次更新请求
            '''
            # time.sleep(2)
            # print('当前时间：',time.strftime('%H-%M-%S',time.localtime(time.time())))
            self.userLog.log.info(f'第{self.round_num}次更新完成')

    def gan_attack(self,cRound):
        if not self.local_model:
            return
        if not self.ganAttackModel:
            self.ganAttackModel=GAN(self.local_model)
        self.ganAttackModel.train(5,512)
        self.ganAttackModel.trainDiscriminator(20,1024)
        if cRound%25==0:
            self.ganAttackModel.sample_images(cRound)

    def membership_attack(self,currWeight):
        if self.local_model is None:
            return
        if self.membershipModel is None:
            self.membershipModel=MATK_MODEL(self.local_model)
        self.membershipModel.update_atk_model(currWeight)

    def ppa_attack(self,currWeight):
        if not self.local_model:
            return
        if not self.ppaAtkModel:
            self.ppaAtkModel=PATK_MODEL(self.local_model)
        self.ppaAtkModel.update_atkModel(currWeight)

    def run(self):
        self.sio.connect(f'http://{self.sHost}:{self.sPort}')
        self.isConnected=True
        self.sio.emit('client_wake_up') # 自动连接
        # self.sio.wait()

    def stop(self):
        if self.isConnected:
            self.sio.disconnect()
            self.userLog=None
            self.isConnected=False

    def obj_to_pickle_string(self,x):
        '''
        python对象转为2进制并序列化为base64
        '''
        return codecs.encode(pickle.dumps(x), "base64").decode()

    def pickle_string_to_obj(self,s):
        '''
        从base64反序列化python对象
        '''
        return pickle.loads(codecs.decode(s.encode(), "base64"))  # 模型返序列化loads，编解码en/decode

# if __name__ == "__main__":
#     client = FederatedClient("127.0.0.1", 5000)
#     client.start()
