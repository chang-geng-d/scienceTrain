import os.path
import keras.backend as K
from keras.datasets import mnist
from dp import add_noise
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def create_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', name="dense_1"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(x_train, y_train, x_test, y_test, epochs, sigma):
    model = create_model()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    if sigma > 0:
        for e in range(epochs):
            print(f"------------------epoch:{e}--------------------")
            weights1 = model.get_weights()
            model.fit(x_train, y_train,
                      batch_size=128,
                      epochs=1,
                      verbose=2,
                      validation_data=(x_test, y_test))
            weights2 = model.get_weights()
            lr = K.get_value(model.optimizer.lr)
            noise_weights = add_noise(weights1, weights2, sigma, lr=lr)
            model.set_weights(noise_weights)
        model.save(f"./dp_mnist_model_weights.h5")
    else:
        for e in range(epochs):
            print(f"------------------epoch:{e}--------------------")
            model.fit(x_train, y_train,
                      batch_size=128,
                      epochs=1,
                      verbose=2,
                      validation_data=(x_test, y_test))
        model.save(f"./mnist_model_weights.h5")


def test_model(x_test, y_test, sigma):
    if sigma > 0:
        model = load_model("./dp_mnist_model_weights.h5")
    else:
        model = load_model("./mnist_model_weights.h5")
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test, 10)
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


def main():
    sigma = 1
    epochs = 50
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_model(x_train, y_train, x_test, y_test, epochs=epochs, sigma=sigma)
    test_model(x_test, y_test, sigma)


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = create_model()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=50,
              verbose=2,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


if __name__ == '__main__':
    main()
    # train()
