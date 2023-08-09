import numpy as np
import keras
import random
import tensorflow as tf
from keras import Sequential,losses
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,Flatten,Dropout
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# def build_model():
#         model = Sequential()
#         model.add(Convolution2D(32, (5, 5), input_shape=(28,28,1)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Convolution2D(32, (5, 5)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Convolution2D(64, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         model.add(Dense(64))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.3))
#         model.add(Dense(10))
#         model.add(Activation('softmax'))
#         return model

class MATK_MODEL:
    def __init__(self,currModel:Sequential):
        self.templateModel=keras.models.clone_model(currModel)
        self.atkModel=self.create_atk_model()
        self.loss=losses.binary_crossentropy
        # 为提速，故将之后所用的有序数据集保存为成员
        (Xtrain,Ytrain),(Xtest,Ytest)=mnist.load_data()
        self.sortedXtrain=self.sort_data(Xtrain,Ytrain)
        self.sortedXtest=self.sort_data(Xtest,Ytest)

    def create_atk_model(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=10))  # 全连接模型
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
        # model.summary()
        return model

    def update_atk_model(self,weight):
        self.templateModel.set_weights(weight)
        # 正负例样本形成
        Xprop,Yprop=self.sample_atk_data()
        Xnoprop,Ynoprop=self.sample_atk_data(dataNo=9)
        Xprop=Xprop.reshape(-1,28,28,1)[:1000]
        Yprop=tf.one_hot(indices=Yprop,depth=10,axis=1).numpy()[:1000,:]
        Xnoprop=Xnoprop.reshape(-1,28,28,1)[:1000]
        Ynoprop=tf.one_hot(indices=Ynoprop,depth=10,axis=1).numpy()[:1000,:]
        # 正负例训练集实现
        trainX=self.get_grads(Xprop,Yprop)
        trainY=np.ones(shape=trainX.shape[0])
        _trainX=self.get_grads(Xnoprop,Ynoprop)
        trainX=np.concatenate((trainX,_trainX),axis=0)
        trainY=np.concatenate((trainY,np.zeros(shape=_trainX.shape[0])),axis=0)
        # batchSize=64
        # # 正例梯度得出
        # stepsPerEpoch=Xprop.shape[0]//batchSize
        # for epoch in range(1):
        #     for step in range(stepsPerEpoch):
        #         idx=np.random.randint(0,Xprop.shape[0],batchSize)
        #         with tf.GradientTape() as tape:
        #             predY=self.templateModel(Xprop[idx,:,:,:],training=True)
        #             loss=self.loss(Yprop[idx,:],predY)
        #         trainAbleVars=self.templateModel.trainable_variables
        #         grads=tape.gradient(loss,trainAbleVars)
        #         if trainX is None:
        #             trainX=grads[-1].numpy().reshape(-1,10)
        #             trainY=np.array([1])
        #         else:
        #             trainX=np.concatenate((trainX,grads[-1].numpy().reshape(-1,10)),axis=0)
        #             trainY=np.concatenate((trainY,np.array([1])),axis=0)
        # 负例梯度得出(此处由于梯度更新未应用故可以重复使用模型)
        # stepsPerEpoch=Xnoprop.shape[0]//batchSize
        # for epoch in range(1):
        #     for step in range(stepsPerEpoch):
        #         idx=np.random.randint(0,Xnoprop.shape[0],batchSize)
        #         with tf.GradientTape() as tape:
        #             predY=self.templateModel(Xnoprop[idx,:,:,:],training=True)
        #             loss=self.loss(Ynoprop[idx,:],predY)
        #         trainAbleVars=self.templateModel.trainable_variables
        #         grads=tape.gradient(loss,trainAbleVars)
        #         trainX=np.concatenate((trainX,grads[-1].numpy().reshape(-1,10)),axis=0)
        #         trainY=np.concatenate((trainY,np.array([0])),axis=0)
        self.atkModel.fit(x=trainX,y=trainY,epochs=5)

    def atk(self,weight):
        self.templateModel.set_weights(weight)
        # 创建测试数据集
        (xTrain,yTrain),(_,_)=mnist.load_data()
        xTrain=xTrain[:1000]/127.5-1
        yTrain=to_categorical(yTrain[:1000],10)
        _sample_list=[i for i in range(xTrain.shape[0])]
        _sample_list=random.sample(_sample_list,8)
        sample_data=xTrain[_sample_list,:]
        sample_data=np.concatenate((sample_data,np.random.normal(0,1,(8,28,28))),axis=0)
        sample_data=sample_data.reshape(-1,28,28,1)
        sample_label=yTrain[_sample_list,:]
        # 攻击
        sample_trueVal=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
        grads=self.get_grads(sample_data,sample_label,batchSize=1)
        sample_attack=np.argmax(self.atkModel.predict(grads),axis=1)

        self.show_atk_result(sample_data,sample_attack,sample_trueVal)
        self.show_grads(grads)

    def get_grads(self,sampleX:np.ndarray,sampleY:np.ndarray,batchSize=64):
        trainX=None
        # 梯度得出
        stepsPerEpoch=sampleX.shape[0]//batchSize
        for epoch in range(1):
            for step in range(stepsPerEpoch):
                idx=np.random.randint(0,sampleX.shape[0],batchSize)
                with tf.GradientTape() as tape:
                    predY=self.templateModel(sampleX[idx,:,:,:],training=True)
                    loss=self.loss(sampleY[idx,:],predY)
                trainAbleVars=self.templateModel.trainable_variables
                grads=tape.gradient(loss,trainAbleVars)
                if trainX is None:
                    trainX=grads[-1].numpy()
                else:
                    trainX=np.concatenate((trainX,grads[-1].numpy()),axis=0)
        return trainX.reshape(-1,10)

    def sample_atk_data(self,dataNo=-1):
        '''
        根据dataNo编号去除数据集中某一类的数据，并返回重构后的类规格化数据集供攻击模型训练
        '''
        tX,tY=None,[]
        for i in range(len(self.sortedXtrain)):
            if i!=dataNo:
                if tX is None:
                    tX=self.sortedXtrain[i]
                else:
                    tX=np.concatenate((tX,self.sortedXtrain[i]),axis=0)
                tY+=[i for j in range(self.sortedXtrain[i].shape[0])]
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

    def show_atk_result(self,data,atkPred,trueVal):
        fig, axs = plt.subplots(4,4)
        fig.suptitle('成员推理攻击示例')
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(data[i*4+j,:,:,0], cmap='gray')
                axs[i, j].axis('off')
                axs[i,j].set_title(f'攻击模型输出:{atkPred[i*4+j]},真实标签:{trueVal[i*4+j]}')
        plt.savefig('static/images/membership_imgs/results.png')
        plt.close()

    def show_grads(self,grads_list):
        rowLabels = ['成员 1','成员 2','成员 3','成员 4','成员 5','成员 6','成员 7','成员 8',
             '非成员1','非成员2','非成员3','非成员4','非成员5','非成员6','非成员7','非成员8']
        colLabels=['0','1','2','3','4','5','6','7','8','9']
        cellTexts = np.around(grads_list.tolist(),4)

        plt.figure(figsize=(20,8))
        plt.title("成员 / 非成员梯度向量")
        plt.xticks(np.arange(len(colLabels)),labels=colLabels)
        plt.yticks(np.arange(len(rowLabels)),labels=rowLabels)

        for i in range(len(rowLabels)):
            for j in range(10):
                text=plt.text(j,i,cellTexts[i,j],ha='center',va='center',color='w')
        plt.imshow(cellTexts,aspect="auto",cmap='coolwarm')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('static/images/membership_imgs/grads.png')
        plt.close()

# if __name__=='__main__':
    # (Xtrain,Ytrain),(Xtest,Ytest)=mnist.load_data()
    # print(Xtrain.shape)
    # print(Ytrain.shape)
    # print(max(np.unique(Ytrain)))

    # Tmodel=build_model()
    # model=MATK_MODEL(Tmodel)
    # model.update_atk_model()
    # tXtrain,tYtrain=model.sample_atk_data()
    # print(tXtrain.shape)
    # print(tYtrain.shape)
    # print(np.unique(tYtrain))