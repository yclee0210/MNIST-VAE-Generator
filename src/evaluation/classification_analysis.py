# Import libraries and modules
import os
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

import config

# Define model architecture
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
model.load_weights("weights.best.hdf5")

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Load pre-shuffled MNIST data into train and test sets
sampling = config.SAMPLE_COUNT
working_dir = config.WORKING_DIR + sampling + "/"

print ("Sampling: ", sampling)
stdev_folder_arr = os.listdir(working_dir)

for stdev_folder in stdev_folder_arr:
    std_dev = stdev_folder.split("_")[1]
    print ("Std Dev: ", std_dev)
    print ("Label, Loss, Accuracy")

    label_folder_arr = os.listdir(working_dir + stdev_folder)
    for label_folder in label_folder_arr:
        if (label_folder == 'samples.png'):
            continue
        label = label_folder.split("_")[1]
        contents = os.listdir(working_dir + stdev_folder + "/" + label_folder)
        for file in contents:
            filepath = stdev_folder + "/" + label_folder + "/" + file
            X_test = np.load(working_dir + filepath)
            y_test = np.zeros(X_test.shape[0],)
            y_test.fill(label)
            # Preprocess input data
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
            X_test = X_test.astype('float32')
            # Preprocess class labels
            Y_test = np_utils.to_categorical(y_test, 10)


            scores = model.evaluate(X_test, Y_test, verbose=0)
            print(label,",",scores[0],",",scores[1])