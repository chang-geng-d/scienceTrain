from __future__ import print_function, division

import os
# from keras.datasets import mnist
from keras.layers import *
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

# class LocalModel:
#     '''
#     模拟本地模型,检验gan攻击模块的可插拔性,仅用于测试
#     对应 server.py 与 client.py 的全局模型与本地模型
#     '''
#     def __init__(self):
#         self.model = Sequential()
#         self.model.add(Convolution2D(32, (5, 5), input_shape=(28,28,1)))    
#         # 需要换为5*5卷积核以将28*28全部卷积以防止生成时的由于未更新导致的边角乱码问题
#         self.model.add(Activation('relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.model.add(Convolution2D(32, (5, 5)))
#         self.model.add(Activation('relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.model.add(Convolution2D(64, (3, 3)))
#         self.model.add(Activation('relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.model.add(Flatten())
#         self.model.add(Dense(64))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(0.3))
#         self.model.add(Dense(10))
#         self.model.add(Activation('softmax'))

#         # self.model.summary()

#         # 由于未知原因，优化器不同也会影响攻击的效果
#         self.model.compile(optimizer=Adam(0.0002, 0.5),loss='binary_crossentropy',metrics=['accuracy'])
#         if os.path.exists('gan_attack/models/discriminator.h5'):    # 测试时为训练可重入节约时间
#             self.model.load_weights('gan_attack/models/discriminator.h5')
    
#     def train(self,sample,t_epochs):
#         '''
#         sample为元组,默认x在[-1,1],y为独热编码
#         t_epochs仅用于模拟服务器每一轮分发
#         '''
#         self.model.trainable=True
#         (X_sample,Y_sample)=sample
#         self.model.fit(X_sample,Y_sample,epochs=t_epochs,batch_size=64,shuffle=True)
#         self.model.trainable=False

class GAN:
    def __init__(self,discriminator):#:LocalModel):
        self.latent_dim = 100
        self.discriminator=discriminator
        self.generator = self.build_generator()
        # if os.path.exists('gan_attack/models/generator.h5'):
        #     self.generator.load_weights('gan_attack/models/generator.h5')
        
        # 建立联合模型
        z = Input(shape=(self.latent_dim,))
        validity = self.discriminator.model(self.generator(z))
        self.combined = Model(inputs=z, outputs=validity)   # 初始化生成器与判别器联合模型
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5),metrics=['accuracy'])

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod((28,28,1)), activation='tanh'))
        model.add(Reshape((28,28,1)))

        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)
    
    def trainDiscriminator(self,fakeEpochs,sampleNum):#,t_epochs):
        '''
        本地模型的训练权还是可被攻击者任意调用的
        t_epochs 测试所用
        '''
        # 使用标识为非推断类的生成数据训练本地模型,逼迫暴露更多特征
        noise=np.random.normal(0, 1, (sampleNum, self.latent_dim))
        X_fake=self.generator.predict(noise)
        Y_fake=np.zeros((sampleNum,10))
        for i in range(sampleNum):
            Y_fake[i,np.random.randint(low=0,high=9)]=1
        # self.discriminator.train((X_fake,Y_fake),fakeEpochs)
        self.discriminator.model.trainable=True
        self.discriminator.model.fit(X_fake,Y_fake,epochs=fakeEpochs,batch_size=64,shuffle=True)
        self.discriminator.model.trainable=False
        
        # # 模拟服务器的合法参与者对模型的纠正训练
        # (X_train, Y_train), (_,_) = mnist.load_data()
        # train_pos=np.random.randint(0,X_train.shape[0],sampleNum)    
        # # 正确训练集和错误训练集的规模差距不能太大才能实现逼迫暴露更多特征的目的
        # X_train=X_train[train_pos]/127.5-1
        # Y_train=tf.one_hot(indices=Y_train[train_pos],depth=10,axis=1)
        # self.discriminator.train((X_train,Y_train),t_epochs)    
        # # 此处epoch仅为了模拟使用，实际上服务器-客户机需要进行t_epochs次的联邦学习

        # # 注意: 此处违反了模块设计准则，仅为了测试时保证时间花销所用
        # self.discriminator.model.save_weights('gan_attack/models/discriminator.h5')

    def train(self,epochs,trainNum):
        '''
        训练生成器
        '''
        noise = np.random.normal(0, 1, (trainNum, self.latent_dim))
        valid=np.zeros((trainNum,10))
        valid[:,-1]=1   # 选定9为要推理内容的类
        self.combined.fit(noise,valid,epochs=epochs,batch_size=64)

        # if times%5==0:
        #     self.sample_images(f'{times}')

        # self.generator.save_weights('gan_attack/models/generator.h5')

    def sample_images(self):
        '''
        保存该次输出的图片
        '''
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 将[-1,1]的输出规整至[0,1]以便显示
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[i*5+j, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
        fig.savefig(f"gan_attack/images/{int(time.time())}.png")
        plt.close()


# if __name__ == '__main__':
#     dis=LocalModel()
#     gan = GAN(dis)

#     # 先对discriminator预训练以模拟已存在的全局模型
#     (X_train,Y_train),(_,_)=mnist.load_data()
#     X_train=X_train/127.5-1
#     Y_train=tf.one_hot(indices=Y_train,depth=10,axis=1)
#     dis.train((X_train,Y_train),5)

#     # 模拟gan攻击,共76轮
#     for i in range(76):
#         gan.train(5,512,i)
#         gan.trainDiscriminator(25,1024,5)   # 小的错误训练集生成是为了加快训练速度   