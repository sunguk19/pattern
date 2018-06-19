# load PINES dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Conv3D, MaxPooling3D, Flatten, Dropout


# this applies load dataset and reshape for keras training.
x_train = np.load("x_train_79.npy")
x_test = np.load("x_test_79.npy")
y_train = np.load("y_train_79.npy")
y_test = np.load("y_test_79.npy")
x_train = x_train.reshape(2687, 79, 95, 68, 1) #1 is channel
x_test = x_test.reshape(1322, 79, 95, 68, 1) #1 is channel
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train[:,1:6]	#1:6 for 1~5 emotional state
y_test = y_test[:,1:6]	#1:6 for 1~5 emotional state

model = Sequential()
# input: 79x95x68 images tensors.
# this applies 32 convolution filters of size 5x5x5 each.
model.add(Conv3D(32, (5, 5, 5), activation='relu', input_shape=(79, 95, 68, 1), padding='valid', data_format="channels_last"))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.5))

# this applies 32 convolution filters of size 3x3x3 each.
model.add(Conv3D(32, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.5))

# this applies flatten.
model.add(Flatten())

# this applies fully connected layers.
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# this applies model compile optimizer adam, calculate 'accuracy, loss'.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=10)

# this applies view neural network summary.
print(model.summary())

# this applies view [loss, accuracy].
print(score)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')

# summarize history for loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')

