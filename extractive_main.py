from untitled import DataPreparation, MailServices
from Models import Models
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt
import time
import keras.backend as K
import math as m
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--target_distance', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=6)
parser.add_argument('--number_of_samples', type=int, default=100)
parser.add_argument('--at_least', type=int, default=100)
args = parser.parse_args()

print("Starting operations for {} samples.".format(args.number_of_samples))
extraction_time = time.time()
selected_points, countries, descriptions = DataPreparation.import_data(7, [1,7,15,21], args.number_of_samples, top_required_position=100, dump_folder="data/sequences/not_scaled/multiples", processors=4, scale=False)
extraction_time = time.time() - extraction_time

print("Operations concluded.")


"""
Best up to now:
	history, learning_time, model_description, model = Models.single_lstm_layer(X_train, X_test, Y_train, Y_test, epochs=600, 
		number_of_cells=800, p_dropout = 0.3, dense_activation = 'scaled_sigmoid', optimizer='adam')
"""
