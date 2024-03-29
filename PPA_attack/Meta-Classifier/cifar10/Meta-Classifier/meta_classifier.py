import time

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.image as img
from keras.utils import to_categorical

def create_attack_model():# 无引用
    '''
    创建10分类的卷积攻击模型，输入为10*896*1
    '''
    input_shape = (10, 896, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=['accuracy'])
    model.summary()
    return model


def create_attack_model2():
    '''
    本质上是10分类的全连接层卷积攻击模型
    '''
    input_shape = (10, 1, 1)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=['accuracy'])
    model.summary()
    return model


def load_data(data_path):
    '''
    从指定的路径中从图片加载攻击模型的训练与测试数据集
    '''
    print("load data...")
    x_train_attack = []
    y_train_attack = []
    x_test_attack = []
    y_test_attack = []
    for i in range(10):
        for j in range(100):
            read_list_path = data_path + f"/train_data/num/{i}/{j}.jpg"
            x_list_temp = img.imread(read_list_path)
            x_train_attack.append(x_list_temp)
            y_train_attack.append(i)
    for i in range(10):
        for j in range(20):
            read_list_path = data_path + f"/test_data/num/{i}/{j}.jpg"
            x_list_temp = img.imread(read_list_path)
            x_test_attack.append(x_list_temp)
            y_test_attack.append(i)
    x_train_attack = np.array(x_train_attack)
    y_train_attack = np.array(y_train_attack)
    x_test_attack = np.array(x_test_attack)
    y_test_attack = np.array(y_test_attack)
    return x_train_attack, y_train_attack, x_test_attack, y_test_attack


def attack(data_path):
    '''
    具体攻击模块
    '''
    t0 = time.time()

    attack_model = create_attack_model2()
    # 加载并变换数据集
    x, y, x_test, y_test = load_data(data_path)
    x_train = x.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 10, 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 10, 1, 1)
    # 使用数据集训练模型
    attack_model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_test, y_test))
    attack_model.save("./result/attack_model.h5")

    t1 = time.time()
    # 测试模型性能
    for i in range(10):
        score = attack_model.evaluate(x_test[i*20:i*20+20], y_test[i*20:i*20+20], verbose=2)
        print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("------------------------------------------------")
    score = attack_model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')

    t2 = time.time()
    ft = open(rf"./result/attack_time.txt", "a+")
    print(f"{path}:  train:{t1 - t0}s, test:{t2 - t1}s", file=ft)
    ft.close()


if __name__ == '__main__':
    path = "./result"
    attack(path)
