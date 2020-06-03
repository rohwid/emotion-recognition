from __future__ import print_function

import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import TensorBoard


def inspect_dataset():
    print("DATASET INFO: ")
    print(df.info())

    print("\nINFO IN \"USAGE\" COLUMN: ")
    print(df["Usage"].value_counts())

    print("\nDATASET HEAD PREVIEW: ")
    print(df.head())

    print("\nVALUES IN \"train_x\": ")
    print(f"{train_x[0:4]}")
    print("LENGTH OF \"train_x\": ", train_x.shape)

    print("\nVALUES IN \"train_y\": ")
    print(f"{train_y[0:4]}")
    print("LENGTH OF \"train_y\": ", len(train_y))

    print("\nVALUES IN \"test_x\": ")
    print(f"{test_x[0:4]}")
    print("LENGTH OF \"test_x\": ", len(test_x))

    print("\nVALUES IN \"test_y\": ")
    print(f"{test_y[0:4]}")
    print("LENGTH OF \"test_y\": ", len(test_y))


def inspect_data():
    print("DATASET HEAD PREVIEW: ")
    print(df.head())

    print("\nVALUES IN \"train_x\": ")
    print(f"{train_x[0:4]}")
    print("LENGTH OF \"train_x\": ", train_x.shape)

    print("\nVALUES IN \"train_y\": ")
    print(f"{train_y[0:4]}")
    print("LENGTH OF \"train_y\": ", train_y.shape)

    print("\nVALUES IN \"test_x\": ")
    print(f"{test_x[0:4]}")
    print("LENGTH OF \"test_x\": ", test_x.shape)

    print("\nVALUES IN \"test_y\": ")
    print(f"{test_y[0:4]}")
    print("LENGTH OF \"test_y\": ", test_y.shape)


df = pd.read_csv('datasets/fer2013.csv')

train_x, train_y, test_x, test_y = [], [], [], []

for index, row in df.iterrows():
    val = row['pixels'].split(" ")

    try:
        if 'Training' in row['Usage']:
            train_x.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])

        if 'PublicTest' in row['Usage']:
            test_x.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f'error occurred at index: {index} and row: {row}')

train_x = np.array(train_x, 'float32')
train_y = np.array(train_y, 'float32')
test_x = np.array(test_x, 'float32')
test_y = np.array(test_y, 'float32')

# inspect_dataset()

# Normalizing the data (the value set between 0 and 1)
train_x -= np.mean(train_x, axis=0)
train_x /= np.std(train_x, axis=0)
test_x -= np.mean(test_x, axis=0)
test_x /= np.std(test_x, axis=0)

# inspect_data()

num_features = 64
num_labels = 7
batch_size = 64
epochs = 60
width, height = 48, 48

train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

train_x = train_x.reshape(train_x.shape[0], width, height, 1)
test_x = test_x.reshape(test_x.shape[0], width, height, 1)

# 1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(train_x.shape[1:])))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

# Fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

model.summary()

callbacks = TensorBoard(log_dir='./graph')

# Compiling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# Training the model
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_x, test_y),
          shuffle=True,
          callbacks=[callbacks])

# Saving the  model to  use it later on
fer_json = model.to_json()

with open("emotion_classification_cnn_5_emotions.json", "w") as json_file:
    json_file.write(fer_json)

model.save_weights("emotion_classification_cnn_5_emotions.h5")
