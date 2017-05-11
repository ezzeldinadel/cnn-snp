import csv
import os
import pandas as pd
import numpy as np
import datetime as dt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam



def load_data():
	path = './data/full.csv'

	# load full data
	D = pd.read_csv(path, index_col=0)
	# D.index = pd.to_datetime(D.index, format='%Y%m%d')

	print D


	# extract inputs/targets
	target_cols = list(filter(lambda x: 'c1_c0' in x, D.columns.values))
	input_cols  = list(filter(lambda x: 'c1_c0' not in x, D.columns.values))
	InputDF = D[input_cols]
	TargetDF = D[target_cols]

	# into numpy format
	InputDF = InputDF.as_matrix()
	TargetDF = TargetDF.as_matrix()

	# 6 values, from 440 symbols, over 2768 days
	InputDF = np.reshape(InputDF, (InputDF.shape[0], 440, 6))

	test_size = 500;
	window_size = 28;

	# X_train = np.ndarray((window_size, 440, 6))
	# X_test = np.ndarray((window_size, 440, 6))
	# for i in range(2768 - (test_size + window_size - 1)):
	# 	X_train = np.append(X_train, matrix[i:i + window_size], axis=0)
	# for i in range(2768 - (test_size + window_size - 1), 2768 - window_size):
	# 	X_test = np.append(X_test, matrix[i:i + window_size], axis=0)

	X_train = np.array([InputDF[i:i + window_size] for i in range(InputDF.shape[0] - (test_size + window_size - 1) )])
	X_test = np.array([InputDF[i:i + window_size] for i in range(InputDF.shape[0] - (test_size + window_size - 1), InputDF.shape[0] - window_size + 1)])


	y_train = TargetDF[window_size - 1 : -test_size]
	y_test = TargetDF[-test_size:]

	# print matrix[:20].shape
	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape


	return X_train, y_train, X_test, y_test


# LOAD ALL DATA
X_train, y_train, X_test, y_test = load_data()

# KERAS STARTS HERE
batch_size = 16
epochs = 10

# dimensions
window_size, tickers, channels = 28, 440, 6
input_shape = (window_size, tickers, channels)

# model definition
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(440))


# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
# print('Test accuracy:', score[1])