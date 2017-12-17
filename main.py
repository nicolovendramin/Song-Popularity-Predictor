from ParallelUtils import DataPreparation, BaselineComputation, MailServices
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
parser.add_argument('--weights_file', type=str, default=None)
parser.add_argument('--target_distance', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=6)
parser.add_argument('--number_of_samples', type=int, default=100)
parser.add_argument('--at_least', type=int, default=100)
args = parser.parse_args()

# 7 , 1, 1500; 7,7,1500 ; 7,15,1500 ; 7,21,1500 
# 7 , 1, 5000; 7,7,5000 ; 7,15,5000 ; 7,21,5000
# 7 , 1, 10000; 7,7,10000 ; 7,15,10000 ;  

extraction_time = time.time()
X_train, Y_train, X_test, Y_test, description = DataPreparation.import_train_test(7, 21, 10000,
 top_required_position=100, sequence_dumps="data/sequences/not_scaled/multiples", splits_dumps="data/splits/not_scaled/multiples", 
 processors=2, scale=False)
extraction_time = time.time() - extraction_time
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

baselines_to_keep = 4
results, baselines, accuracies = BaselineComputation.compute_baselines(X_train, Y_train, X_test, Y_test)

for i in range(0, len(results)):
	print("The {} baseline scores a mae of {:.2f} and an accuracy of {:.2f}.".format(baselines[i], results[i], accuracies[i]))

show = "email"

history, learning_time, model_description, model = Models.single_lstm_layer(X_train, X_test, Y_train, Y_test, epochs=300, 
	number_of_cells=800, p_dropout = 0.3, dense_activation = 'scaled_sigmoid', optimizer='adam',model_file=None, weights_file="1512530339.h5")
#history, learning_time, model_description, model = Models.single_layer_rnn(X_train, X_test, Y_train, Y_test, epochs=600, 
#	number_of_cells=800, p_dropout = 0.3, dense_activation = 'scaled_sigmoid', optimizer='adam')

# serialize model to JSON
add_on = ""

try:
	filename = str(int(time.time()))
	model_json = model.to_json()
	json_file = open("tmp/models/" + filename + ".json", "w+")
	json_file.write(model_json)
	json_file.close()
	add_on += "Saved the model in file {}[.json].\n".format(filename)
except Exception as e:
	add_on += "Couldn't dump the model.\n"

try:
	model.save_weights("tmp/weights/" + filename + ".h5")
	add_on += "Saved the weights in file {}[.h5].\n".format(filename)
except Exception as e:
	add_on += "Couldn't dump the weights.\n"
	
print(add_on)

for i in range(0, len(results)):
	print("The {} baseline scores a mae of {:.2f} and an accuracy of {:.2f}.".format(baselines[i], results[i], accuracies[i]))

model_prediction = model.predict(X_test)
acc = BaselineComputation.accuracy_computer(model_prediction, X_test, Y_test, "trained model")

# plot history
lss = ['-', ':', '-.', '--']
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
for i in sorted(zip(baselines, results), key=lambda x: x[1])[:baselines_to_keep]:
	plt.plot(np.full((len(history.history['val_loss']), 1), i[1]), label=i[0], ls=lss.pop())
plt.legend()
plt.savefig("tmp/loss.png")
if show == "display":
	plt.show()
plt.clf()

last_train_loss = history.history['loss'][-100:]
last_test_loss = history.history['val_loss'][-100:]
baselines.append("trained model")
results.append(history.history['val_loss'][-1])
accuracies.append(acc)
baseline_comparisons = ""
for i in range(0, len(results)):
	baseline_comparisons += "The {} baseline scores a mae of {:.2f} and an accuracy of {:.2f}.\n".format(baselines[i], results[i], accuracies[i])

msg = model_description
msg += "[" + description + "]\n"
msg += "Final train loss (avg over last {} epochs) : {:.2f},\n".format(len(last_train_loss), float(sum(last_train_loss))/len(last_train_loss))
msg += "Final test loss (avg over last {} epochs) : {:.2f},\n".format(len(last_train_loss), float(sum(last_test_loss))/len(last_test_loss))
msg += "Data extraction time : {:.2f}s [{:.2f} m],\n".format(extraction_time, extraction_time/60)
msg += "Learning time : {:.2f}s [{:.2f} m],\n".format(learning_time, learning_time/60)
msg += add_on
msg += baseline_comparisons
if show == "email":
	MailServices.send_mail([msg], ["tmp/loss.png"])
 
file = open("tmp/final_results.txt", 'a')
file.write("-------------------------------\n" + msg + "\n-------------------------------")
file.close()

print(msg)
print("Operations concluded.")


"""
Best up to now:
	history, learning_time, model_description, model = Models.single_lstm_layer(X_train, X_test, Y_train, Y_test, epochs=600, 
		number_of_cells=800, p_dropout = 0.3, dense_activation = 'scaled_sigmoid', optimizer='adam')
"""
