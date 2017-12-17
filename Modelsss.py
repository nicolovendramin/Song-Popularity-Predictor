import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import time
import keras.backend as K
import math as m
from sklearn import linear_model
from keras.utils.generic_utils import get_custom_objects

class Models:

	def custom_activation(x):
		return (K.sigmoid(x) * 201) - 1







	@staticmethod
	def single_lstm_layer(X_train, X_test, Y_train, Y_test, number_of_cells=800, p_dropout = 0.2, dense_activation = 'linear', epochs=1000, optimizer='adam', weights_file=None, model_file=None):
		learning_time = time.time()
		print(X_train[0])
		model = Sequential()
		model.add(LSTM(number_of_cells), input_shape=(1, 7))
		model.add(Dropout(p_dropout))
		model.add(Dense(1))
		model.add(Activation(dense_activation))

		if model_file != None:
			try:
				json_file = open("tmp/models/" + model_file, 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				model_ = model_from_json(loaded_model_json)
				model = model_
				print("Loaded model from file {}[.json].".format(model_file))
			except Exception as e:
				print("Model File not found")

		if weights_file != None:
			try:
				model.load_weights("tmp/weights/" + weights_file)
				print("Loaded weights from file {}[.h5].".format(weights_file))
			except Exception as e:
				print("Weights File not found")

		model.compile(loss='mae', optimizer=optimizer)
		# fit network
		history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
		learning_time = time.time() - learning_time

		model_description = "Single LSTM layer with {} cells, dropout probability of {} and a {} activation in the dense layer. Total {} epochs with {} optimizer. \n".format(number_of_cells, p_dropout, dense_activation, epochs, optimizer)

		return history, learning_time, model_description, model


	@staticmethod
	def multiple_lstm_layers(X_train, X_test, Y_train, Y_test, number_of_cells=800, p_dropout = 0.2, dense_activation = 'linear', epochs=1000, weights_file=None, model_file=None):
		learning_time = time.time()
		model = Sequential()
		model.add(LSTM(int(number_of_cells/2), input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
		model.add(LSTM(int(number_of_cells/4), input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
		model.add(LSTM(int(number_of_cells/4), input_shape=(X_train.shape[1], X_train.shape[2])))
		model.add(Dropout(p_dropout))
		model.add(Dense(54))
		model.add(Activation(dense_activation))

		if model_file != None:
			try:
				json_file = open("tmp/models/" + model_file, 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				model_ = model_from_json(loaded_model_json)
				model = model_
				print("Loaded model from file {}[.json].".format(model_file))
			except Exception as e:
				print("Model File not found")

		if weights_file != None:
			try:
				model.load_weights("tmp/weights/" + weights_file)
				print("Loaded weights from file {}[.h5].".format(weights_file))
			except Exception as e:
				print("Weights File not found")

		model.compile(loss='mae', optimizer='adam')
		# fit network
		history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
		learning_time = time.time() - learning_time

		model_description = "Triple LSTM layer with {} cells, dropout probability of {} and a {} activation in the dense layer. Total {} epochs. \n".format(number_of_cells, p_dropout, dense_activation, epochs)

		return history, learning_time, model_description, model


	@staticmethod
	def single_layer_rnn(X_train, X_test, Y_train, Y_test, number_of_cells=800, p_dropout = 0.2, dense_activation = 'linear', epochs=1000, optimizer='adam', weights_file=None, model_file=None):
		learning_time = time.time()
		model = Sequential()
		model.add(LSTM(number_of_cells, input_shape=(X_train.shape[1], X_train.shape[2])))
		model.add(Dropout(p_dropout))
		model.add(Dense(54))
		model.add(Activation(dense_activation))

		if model_file != None:
			try:
				json_file = open("tmp/models/" + model_file, 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				model_ = model_from_json(loaded_model_json)
				model = model_
				print("Loaded model from file {}[.json].".format(model_file))
			except Exception as e:
				print("Model File not found")

		if weights_file != None:
			try:
				model.load_weights("tmp/weights/" + weights_file)
				print("Loaded weights from file {}[.h5].".format(weights_file))
			except Exception as e:
				print("Weights File not found")

		model.compile(loss='mae', optimizer=optimizer)
		# fit network
		history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
		learning_time = time.time() - learning_time

		model_description = "Single LSTM layer with {} cells, dropout probability of {} and a {} activation in the dense layer. Total {} epochs with {} optimizer. \n".format(number_of_cells, p_dropout, dense_activation, epochs, optimizer)

		return history, learning_time, model_description, model


	@staticmethod
	def add_new_activation_function(activation_function, name):
		get_custom_objects().update({name: Activation(activation_function)})


get_custom_objects().update({'scaled_sigmoid': Activation(Models.custom_activation)})