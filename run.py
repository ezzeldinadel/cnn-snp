import csv
import os
import pandas as pd
import numpy as np
import datetime as dt



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
	matrix = InputDF.as_matrix()


	# 6 values, from 440 symbols, over 2768 days
	matrix = np.reshape(matrix, (matrix.shape[0], 440, 6))

	test_size = 500;
	window_size = 28;

	# X_train = np.ndarray((window_size, 440, 6))
	# X_test = np.ndarray((window_size, 440, 6))
	# for i in range(2768 - (test_size + window_size - 1)):
	# 	X_train = np.append(X_train, matrix[i:i + window_size], axis=0)
	# for i in range(2768 - (test_size + window_size - 1), 2768 - window_size):
	# 	X_test = np.append(X_test, matrix[i:i + window_size], axis=0)

	X_train = np.array([matrix[i:i + window_size] for i in range(2768 - (test_size + window_size - 1) )])
	X_test = np.array([matrix[i:i + window_size] for i in range(2768 - (test_size + window_size - 1), 2768 - window_size + 1)])

	# print matrix[:20].shape
	print X_train.shape
	print X_test.shape

	# y_train = 
	# y_test = 

	return X_train, X_test

d = load_data()