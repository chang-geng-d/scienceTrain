import keras
import numpy as np
import tensorflow as tf
from keras import Sequential,losses,optimizers
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten,Dropout
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

def build_model():
        model = Sequential()
        model.add(Convolution2D(32, (5, 5), input_shape=(28,28,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model

class PATK_MODEL:
    def __init__(self,model):
        self.templateModel=keras.models.clone_model(model)
        self.atkModel=self.create_atk_model()
        self.loss=losses.binary_crossentropy

    def create_atk_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(10, 800, 1)))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=['accuracy'])
        # model.summary()
        return model

    def atk(self,currWeight):
        pass

    def update_atkModel(self,currWeight):
        # 将本地数据集变换形式
        localX,localY=self.get_atk_datasets()
        localX=localX.reshape((-1,28,28,1))
        localY=to_categorical(localY,10)
        # 计算敏感度矩阵
        _,g_be=self.get_inputs(currWeight)
        self.templateModel.reset_states()
        self.templateModel.compile(optimizer=optimizers.Adam(0.0002, 0.5),loss=self.loss,metrics=['accuracy'])
        self.templateModel.fit(localX, localY, epochs=5, batch_size=64, shuffle=True)
        _,g=self.get_inputs()
        Xtrain,Ytrain=self.get_sensitive(g_be,g)
        # 训练分类模型
        Xtrain = Xtrain.astype('float32')
        Xtrain/= 255
        Ytrain = to_categorical(Ytrain, 10)
        Xtrain = Xtrain.reshape(Xtrain.shape[0], 10, 800, 1)
        self.atkModel.fit(Xtrain, Ytrain, epochs=250, batch_size=128)

    def get_sensitive(self,tShadow,tRetrain):
        x_a_n = []
        for item_a, item_n in zip(tShadow, tRetrain):
            a_temp = np.array(item_a)
            n_temp = np.array(item_n)
            temp = a_temp - n_temp
            x_a_n.append(temp)
        x_a_n=np.array(x_a_n)
        Xtrain,Ytrain=[],[]
        for i,item in enumerate(x_a_n):
            Xtrain.append(item)
            Ytrain.append(i)
        return np.array(Xtrain),np.array(Ytrain)

    def get_inputs(self,weights=None):
        if weights is not None:
            self.templateModel.set_weights(weights)
        # 构建原始影子数据集
        Xtrain,Ytrain=self.get_shw_datasets()
        Xtrain=Xtrain.reshape((-1,28,28,1))
        Ytrain=to_categorical(Ytrain,10)
        # 得到影子梯度矩阵
        gradSum,gradList=[],[]
        for i in range(100):
            tSum,tGrad=[],[]
            for j in range(10):
                b=j*1000+8*i
                grads=self.get_grads(Xtrain[b:b+128],Ytrain[b:b+128],128)
                tSum.append(np.sum(grads))
                tGrad.append(grads)
            tSum=np.array(tSum).reshape((10,1))
            tGrad=np.array(tGrad)
            gradSum.append(tSum)
            gradList.append(tGrad)
        return gradSum,gradList

    def get_grads(self,x,y,batchSize=64):
        trainX=None
        # 梯度得出
        stepsPerEpoch=x.shape[0]//batchSize
        for epoch in range(1):
            for step in range(stepsPerEpoch):
                idx=np.random.randint(0,x.shape[0],batchSize)
                with tf.GradientTape() as tape:
                    predY=self.templateModel(x[idx,:,:,:],training=True)
                    loss=self.loss(y[idx,:],predY)
                trainAbleVars=self.templateModel.trainable_variables
                grads=tape.gradient(loss,trainAbleVars)
                if trainX is None:
                    trainX=grads[0].numpy()
                else:
                    trainX=np.concatenate((trainX,grads[0].numpy()),axis=0)
        return trainX.reshape(-1,800)

    def get_atk_datasets(self):
        Xtrain,Ytrain,_,_=mnist.load_data()
        Xsort=self.sort_data(Xtrain,Ytrain)
        sampleNum=200
        tX,tY=None,[]
        for i in range(len(Xsort)):
            if tX is None:
                sampleNum=3000
                tX=Xsort[i][:sampleNum]
            else:
                sampleNum=200
                tX=np.concatenate((tX,Xsort[i][:sampleNum]),axis=0)
            tY+=[i for j in range(sampleNum)]
        return tX,np.array(tY)

    def get_shw_datasets(self):
        Xtrain,Ytrain,_,_=mnist.load_data()
        Xsort=self.sort_data(Xtrain,Ytrain)
        sampleNum=200*5
        tX,tY=None,[]
        for i in range(len(Xsort)):
            if tX is None:
                tX=Xsort[i][:sampleNum]
            else:
                tX=np.concatenate((tX,Xsort[i][:sampleNum]),axis=0)
            tY+=[i for j in range(sampleNum)]
        return tX,np.array(tY)

    def sort_data(self,tX,tY):
        '''
        将原始数据集按标签排序，返回 标签个数*n 大小的二维数组，其中n的大小在每一行可变
        '''
        Ypos=np.argsort(tY)
        tX=tX[Ypos]/127.5-1
        tY=tY[Ypos]
        Ypos=np.bincount(tY)
        Xsort=[]
        pos=0
        for i in range(Ypos.shape[0]):
            Xsort.append(tX[pos:pos+Ypos[i]])
            pos+=Ypos[i]
        return Xsort 
