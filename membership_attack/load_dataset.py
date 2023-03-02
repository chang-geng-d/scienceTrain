from keras.datasets import mnist, cifar10
import cv2


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    for i in range(len(x_train)):
        # y_train[i] == [] ???
        cv2.imwrite(f"./cifar10/data/train/{y_train[i][0]}/{y_train[i][0]}_{i}.jpg", x_train[i])
    for j in range(len(x_test)):
        cv2.imwrite(f"./cifar10/data/test/{y_test[j][0]}/{y_test[j][0]}_{j}.jpg", x_test[j])


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for i in range(len(x_train)):
        print(y_train[i])
        cv2.imwrite(f"./mnist/data/train/{y_train[i]}/{y_train[i]}_{i}.jpg", x_train[i])
    for j in range(len(x_test)):
        cv2.imwrite(f"./mnist/data/test/{y_test[j]}/{y_test[j]}_{j}.jpg", x_test[j])


if __name__ == '__main__':
    # load_mnist()
    load_cifar10()
