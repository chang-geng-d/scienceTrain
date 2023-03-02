from keras.datasets import mnist, cifar10
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import random
import numpy as np


def random_sample(sample, x, y):
    '''
    以sample为下标截取x,y
    '''
    x_sample = []
    y_sample = []
    for s in sample:
        x_sample.append(x[s])
        y_sample.append(y[s])
    x_sample = np.array(x_sample)
    y_sample = np.array(y_sample)
    return x_sample, y_sample


class Mnist_Model:
    def __init__(self, shadow_model_num=10, model_mode="dense", ex=0, attack_mode=1):
        self.img_shape = (28, 28, 1)
        self.num_classes = 10
        if model_mode == "cnn":
            self.target_model = self.create_target_model_cnn()
            self.shadow_model = []
            for num in range(shadow_model_num):
                self.shadow_model.append(self.create_shadow_model_cnn())
        else:
            self.target_model = self.create_target_model_dense()
            self.shadow_model = []
            for num in range(shadow_model_num):
                self.shadow_model.append(self.create_shadow_model_dense())
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # 目标模型训练集20000
        self.target_x_train = x_train[:20000]
        self.target_y_train = y_train[:20000]
        # 目标模型测试集20000
        self.target_x_test = x_train[20000:40000]
        self.target_y_test = y_train[20000:40000]

        if attack_mode:
            # 影子模型训练集随机10000, 测试集随机10000
            self.shadow_x_train = []
            self.shadow_y_train = []
            self.shadow_x_test = []
            self.shadow_y_test = []
            for num in range(shadow_model_num):
                n_train = random.sample(range(40000, 60000), 10000)
                shadow_x_train, shadow_y_train = random_sample(n_train, x_train, y_train)
                n_test = set(range(40000, 60000)) - set(n_train)
                n_test = list(n_test)
                shadow_x_test, shadow_y_test = random_sample(n_test, x_train, y_train)
                with open(f"./mnist/configs/config_{ex}.txt", "a+") as f:
                    print(n_train, file=f)
                    print(n_test, file=f)
                self.shadow_x_train.append(shadow_x_train)
                self.shadow_y_train.append(shadow_y_train)
                self.shadow_x_test.append(shadow_x_test)
                self.shadow_y_test.append(shadow_y_test)

        # 测试集
        self.x_test = x_test
        self.y_test = y_test

    def create_target_model_dense(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model

    def create_shadow_model_dense(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model

    def create_target_model_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model

    def create_shadow_model_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model


class Cifar10_Model:
    def __init__(self, shadow_model_num=10, model_mode="dense", ex=0, attack_mode=1):
        self.img_shape = (32, 32, 3)
        self.num_classes = 10
        if model_mode == "cnn":
            self.target_model = self.create_target_model_cnn()
            self.shadow_model = []
            for num in range(shadow_model_num):
                self.shadow_model.append(self.create_shadow_model_cnn())
        else:
            self.target_model = self.create_target_model_dense()
            self.shadow_model = []
            for num in range(shadow_model_num):
                self.shadow_model.append(self.create_shadow_model_dense())
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # 目标模型训练集15000
        self.target_x_train = x_train[:15000]
        self.target_y_train = y_train[:15000]
        # 目标模型测试集15000
        self.target_x_test = x_train[15000:30000]
        self.target_y_test = y_train[15000:30000]

        if attack_mode:
            # 影子模型训练集随机10000, 测试集随机10000
            self.shadow_x_train = []
            self.shadow_y_train = []
            self.shadow_x_test = []
            self.shadow_y_test = []
            for num in range(shadow_model_num):
                n_train = random.sample(range(30000, 50000), 10000)
                shadow_x_train, shadow_y_train = random_sample(n_train, x_train, y_train)
                n_test = set(range(30000, 50000)) - set(n_train)
                n_test = list(n_test)
                shadow_x_test, shadow_y_test = random_sample(n_test, x_train, y_train)
                with open(f"./cifar10/configs/config_{ex}.txt", "a+") as f:
                    print(n_train, file=f)
                    print(n_test, file=f)
                self.shadow_x_train.append(shadow_x_train)
                self.shadow_y_train.append(shadow_y_train)
                self.shadow_x_test.append(shadow_x_test)
                self.shadow_y_test.append(shadow_y_test)

        # 测试集
        self.x_test = x_test
        self.y_test = y_test

    def create_target_model_dense(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model

    def create_shadow_model_dense(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy'])
        model.summary()
        return model

    def create_target_model_cnn(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         input_shape=self.img_shape,
                         activation='relu',
                         padding='same',
                         name="conv2d_1"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_2"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=128,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_3"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_4"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1024, activation='relu', name="dense_1"))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax', name="dense_2"))

        model.compile(loss="categorical_crossentropy",
                      optimizer="Adam",
                      metrics=['accuracy'])
        model.summary()
        return model

    def create_shadow_model_cnn(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         input_shape=self.img_shape,
                         activation='relu',
                         padding='same',
                         name="conv2d_1"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_2"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=128,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_3"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu',
                         name="conv2d_4"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1024, activation='relu', name="dense_1"))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax', name="dense_2"))

        model.compile(loss="categorical_crossentropy",
                      optimizer="Adam",
                      metrics=['accuracy'])
        model.summary()
        return model


if __name__ == '__main__':
    pass
