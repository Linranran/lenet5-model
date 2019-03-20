import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


model = Sequential()
model.add(keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(5,5), filters=20, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))

model.add(keras.layers.Conv2D(kernel_size=(5,5), filters=50, activation='relu', padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))

# model.add(keras.layers.Flatten())
model.add(keras.layers.Reshape((1800,)))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.save("simple_lenet.h5")


print('Training')
model.fit(X_train, y_train, epochs=1, batch_size=128)


print('Testing')
loss, accuracy = model.evaluate(X_test, y_test)


print('loss', loss)
print('accuracy', accuracy)