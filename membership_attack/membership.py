import os
import sys

from keras import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from models import Mnist_Model, Cifar10_Model, random_sample
import numpy as np
from keras.models import load_model
from keras.datasets import mnist, cifar10

import sys

# savedStdout = sys.stdout 
# print_log = open('log.txt',"w")
# sys.stdout = print_log

def train_model(mode, model, img_shape, x_train, y_train, x_test, y_test, num=0, epochs=100):
    '''
    mode: 'shadow' 'target'
    '''
    # 变换数据集
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)   # 10类别的向量
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], img_shape[0], img_shape[1], img_shape[2])) # 变换为输入大小
    x_test = x_test.reshape((x_test.shape[0], img_shape[0], img_shape[1], img_shape[2]))
    # 训练模型并保存
    if mode=='target':
        epochs=epochs*50
    model.fit(x_train, y_train,batch_size=128,epochs=epochs,verbose=2,validation_data=(x_test, y_test))
    if not os.path.exists(f"./{dataset}/models/ex_{ex}"):   # 没有模型保存目录则创建
        os.makedirs(f"./{dataset}/models/ex_{ex}") 
    model.save(f"./{dataset}/models/ex_{ex}/{mode}_model_weights_{num}.h5") # 保存模型
    # 使用训练好的模型对训练集与测试集的标签分别预测
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    x_train_pre = model.predict(x_train)    #模型分别对训练集和测试集的标签预测
    x_test_pre = model.predict(x_test)
    return x_train_pre, x_test_pre


# 采用10 * 1格式输入
def process_pre(x):
    '''
    将10*1向量的最小的7个元素置0
    '''
    x_copy = np.array(x)    # 复制x
    # 从小到大排序取前7个置零
    # 但不打乱原有数组元素的相对顺序
    index = np.argsort(x_copy)[:7]  
    for i in index:
        x_copy[i]=0
    return x_copy


def load_attack_data(train_0, train_1, test_0, test_1):
    '''
    将标记不同的训练集和测试集分开以打上标记
    '''
    x_train = []
    y_train = []
    for i in range(len(train_0)):   # 处理第i个影子模型
        for tr0, tr1 in zip(train_0[i], train_1[i]):
            x_train.append(process_pre(tr0))    # 将第i个影子模型的训练集数据标签设为1,测试集标签设为0
            y_train.append(0)
            x_train.append(process_pre(tr1))
            y_train.append(1)

    x_test = []
    y_test = []
    for te0, te1 in zip(test_0, test_1):    # 以目标模型的输出作为攻击模型的测试集
        x_test.append(process_pre(te0))
        y_test.append(0)
        x_test.append(process_pre(te1))
        y_test.append(1)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def create_attack_model():
    '''
    创建一个10输入2输出的全连接模型
    '''
    input_shape=10
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=input_shape))  # 全连接模型
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
    model.summary()
    return model


def attack(model):
    if attack_mode:
        # 重训练
        attack_test_1, attack_test_0 = train_model("target",
                                                   model.target_model,
                                                   model.img_shape,
                                                   model.target_x_train,
                                                   model.target_y_train,
                                                   model.target_x_test,
                                                   model.target_y_test)# 以目标模型的输出作为测试集
        attack_train_0 = []
        attack_train_1 = []
        for num in range(shadow_model_num): # 训练影子模型
            attack_train_1_num, attack_train_0_num = train_model("shadow",
                                                                 model.shadow_model[num],
                                                                 model.img_shape,
                                                                 model.shadow_x_train[num],
                                                                 model.shadow_y_train[num],
                                                                 model.shadow_x_test[num],
                                                                 model.shadow_y_test[num],
                                                                 num)
            attack_train_0.append(attack_train_0_num)   # 为多个10*1向量组成的列表的列表
            attack_train_1.append(attack_train_1_num)
    else:
        # 加载现有配置
        # 目标模型训练集和测试集固定
        target_model = load_model(f"./{dataset}/models/ex_{ex}/target_model_weights_0.h5")
        x_train_t = model.target_x_train
        x_test_t = model.target_x_test
        x_train_t = x_train_t.astype('float32')
        x_test_t = x_test_t.astype('float32')
        x_train_t /= 255
        x_test_t /= 255
        x_train_t = x_train_t.reshape((x_train_t.shape[0], model.img_shape[0], model.img_shape[1], model.img_shape[2]))
        x_test_t = x_test_t.reshape((x_test_t.shape[0], model.img_shape[0], model.img_shape[1], model.img_shape[2]))
        attack_test_1 = target_model.predict(x_train_t)
        attack_test_0 = target_model.predict(x_test_t)
        # 影子模型训练集和测试集随机可变，需加载配置文件
        with open(f"./{dataset}/configs/config_{ex}.txt", "r") as f:
            config_shadow_dataset = f.readlines()
        if dataset == "mnist":
            (x_train_s, y_train_s), (_, _) = mnist.load_data()
        elif dataset == "cifar10":
            (x_train_s, y_train_s), (_, _) = cifar10.load_data()
        # 默认cifar10
        else:
            (x_train_s, y_train_s), (_, _) = cifar10.load_data()
        x_train_s = x_train_s.astype('float32')
        x_train_s /= 255
        x_train_s = x_train_s.reshape((x_train_s.shape[0], model.img_shape[0], model.img_shape[1], model.img_shape[2]))
        attack_train_0 = []
        attack_train_1 = []
        for num in range(shadow_model_num):
            n_train = eval(config_shadow_dataset[num * 2])
            n_test = eval(config_shadow_dataset[num * 2 + 1])
            shadow_x_train, shadow_y_train = random_sample(n_train, x_train_s, y_train_s)
            shadow_x_test, shadow_y_test = random_sample(n_test, x_train_s, y_train_s)
            shadow_model = load_model(f"./{dataset}/models/ex_{ex}/shadow_model_weights_{num}.h5")
            attack_train_1_num = shadow_model.predict(shadow_x_train)
            attack_train_0_num = shadow_model.predict(shadow_x_test)
            attack_train_1.append(attack_train_1_num)
            attack_train_0.append(attack_train_0_num)

    # attack_train 影子模型列表数据(依次保存每一个影子模型的predict结果)， attack_test 目标模型数据
    x_train, y_train, x_test, y_test = load_attack_data(attack_train_0, attack_train_1, attack_test_0, attack_test_1)
    # x_train为n*16矩阵
    attack_model = create_attack_model()
    y_train = to_categorical(y_train, 2)    # 将逻辑值y独热编码为n*2向量,n为y原先长度(就是01编码)
    y_test = to_categorical(y_test, 2)
    attack_model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=50,
                     verbose=2,
                     validation_data=(x_test, y_test))  # 通过影子模型提供训练数据以构造01分类的攻击模型
    score = attack_model.evaluate(x_test, y_test, verbose=2)
    print(f'Attack Test loss:{score[0]}, Attack Test accuracy:{score[1]}')


# cifar10 + cnn 攻击效果最好(目前测试最高69%左右), 影子模型越多越好, 攻击模型结构可优化调整. 目标模型和影子模型的训练也可优化调整
if __name__ == '__main__':
    # 攻击模式  1: 重训练 0: 加载现有配置
    attack_mode = 0
    # 第i次实验
    ex = 6
    # 加载数据集
    dataset = "mnist"
    # 影子模型数量
    shadow_model_num = 50
    # 加载模型结构
    model_mode = "dense"
    if attack_mode: # 记录一下日志
        with open("./实验说明.txt", "a+") as f:
            f.seek(0)
            conf = f.readlines()
            if len(conf) == ex - 1:
                print(f"ex_{ex}: dataset:{dataset}, shadow_model_num:{shadow_model_num}, model_mode:{model_mode}", file=f)
            else:
                print("ex num error !")
                print("please set configs correctly !")
                sys.exit()
    else:
        with open("./实验说明.txt", "r") as f:
            conf = f.readlines()
            if conf[ex - 1] == f"ex_{ex}: dataset:{dataset}, shadow_model_num:{shadow_model_num}, model_mode:{model_mode}\n":
                pass
            else:
                print("configs set error !")
                print("please set configs correctly !")
                sys.exit()
    if dataset == "mnist":
        mnist_model = Mnist_Model(shadow_model_num=shadow_model_num, model_mode=model_mode, ex=ex,
                                  attack_mode=attack_mode)
        attack(mnist_model) # 攻击入口函数在此
    elif dataset == "cifar10":
        cifar10_model = Cifar10_Model(shadow_model_num=shadow_model_num, model_mode=model_mode, ex=ex,
                                      attack_mode=attack_mode)
        attack(cifar10_model)
    else:
        print("please choose dataset correctly !")
        sys.exit()
