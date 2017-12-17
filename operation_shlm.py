import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import random
import time
import argparse
import pickle
from progressbar import ProgressBar
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
"""
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
"""

pbar = ProgressBar()

parser = argparse.ArgumentParser()
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--target_distance', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=6)
parser.add_argument('--number_of_samples', type=int, default=100)
parser.add_argument('--at_least', type=int, default=100)
args = parser.parse_args()

def import_data(input_length, target_distance, number_of_samples, top_required_position=200):
    # This function returns a dataset including a number of evolunetions for a song in the national charts among all the
    # countries during a number (input_length) of days, and the ranking for the same song in all the charts some
    # (target_distance) days after. The last parameter is the number of elements in the set. Default is extracting all
    # the possible series.

    # opening the file and reading the data
    data = pd.read_csv('data/data.csv')

    #print(data.head(10))
    # Extracting the countries present in the dataset
    countries = data['Region']
    countries = list(set(countries))
    # Mapping each country to an array position and viceversa
    id_to_country = dict(enumerate(countries))
    country_to_id = {v: k for k, v in id_to_country.items()}

    # Extracting the limit position of the dataset
    worst_position = data['Position'].max(axis=0)

    # Extracting the limit dates of the dataset
    max_date = data['Date'].max(axis=0)
    min_date = data['Date'].min(axis=0)

    # computes the total number of days and the "last useful day to start a sequence nowing the number of
    #  days more we need
    time_delta = dt.datetime.strptime(max_date, '%Y-%m-%d').date() - \
                 dt.datetime.strptime(min_date, '%Y-%m-%d').date()
    max_offset = time_delta.days - input_length - target_distance - 1

    # Starts selecting the samples iterating of a cycle of size number_of_samples
    selected_points = []
    for n in pbar(range(0, number_of_samples)):
        # Randomly selects the day to start the sequence respecting the boundary of having a complete sequence
        starting_date = dt.datetime.strptime(min_date, '%Y-%m-%d').date() + dt.timedelta(np.random.randint(0, max_offset))

        # filters the songs that were on chart that day, with at list a certain rank (top_required_position)
        daily = data['Date'] == str(starting_date)
        condition = data['Position'] < top_required_position

        # We store all the track, author pairs for the songs present in that day
        daily_songs = []
        daily_songs_author_pairs = data[daily & condition][['Track Name', 'Artist']]
        for index, row in daily_songs_author_pairs.iterrows():
            daily_songs.append((row['Track Name'], row['Artist']))

        # We extract a random song from the ones on chart in that day
        daily_songs = list(set(daily_songs))
        daily_num = len(daily_songs)

        # We check that there are enough daily song to randomly draw one
        if(daily_num > 0):
            song_number = np.random.randint(0, daily_num)
            song = daily_songs[song_number]

            #print("song")
            #print(song[:10])

            # Now we start fetching the "international vector" for the selected song in the input days
            inputs = []
            for i in range(0, input_length):
                # We go day by day from the starting date, and we inizialise our vector of position in country
                # as all -1 being -1 the encoding for "not present"
                input_array = np.full(len(countries), float(-1))
                day =  starting_date + dt.timedelta(i)

                # We get from the data the positio of the given song in the date into account
                daily = data['Date'] == str(day)
                author = data['Artist'] == song[1]
                track_name = data['Track Name'] == song[0]

                result = data[daily & author & track_name][['Position', 'Region']]

                # We iterate on the rows of the result and we fill in the position in country vectors in all
                # the meaningful positions (the position is substitued with the distance from the worst position so that
                # the higher is the position the lower the score)
                for index, row in result.iterrows():
                    input_array[country_to_id[row['Region']]] = worst_position - row['Position']

                # We append the input to the set of inputs for this sequence, scaled
                inputs.append(rescaling(input_array, (-1, 200), (-1,1)))

            # we fetch the target as one of the inputs
            target = np.full(len(countries), float(-1))

            day =  starting_date + dt.timedelta(input_length + target_distance - 1)

            daily = data['Date'] == str(day)
            author = data['Artist'] == song[1]
            track_name = data['Track Name'] == song[0]

            result = data[daily & author & track_name][['Position', 'Region']]
            for index, row in result.iterrows():
                target[country_to_id[row['Region']]] = row['Position']

            # We append the pair (list of inputs [position in country array for the considered time window]
            # -> target) to the list of selected points
            inpt = inputs[0]
            for i in range(1, len(inputs)):
                inpt = np.vstack((inpt, inputs[i]))

            #print("input")
            #print(inpt[:10])
            #print("target")
            #print(target[:10])


            selected_points.append((inpt, rescaling(target, (-1, 200), (-1,1))))

        # In case the draw was not ok, we add one more to be done and we skip this iteration
        else:
            print "daily songs number too low:", daily_num
            number_of_samples = number_of_samples + 1

    return selected_points, len(countries)


def rescaling(input_vec, old_interval, new_interval):
    # rescales the input vector in the new interval

    multiplier = float(new_interval[1] - new_interval[0]) / float((old_interval[1] - old_interval[0]))

    size = len(input_vec)
    for i in range(0, size):
        figure = input_vec[i]
        figure = float(figure)
        offset = figure - float(old_interval[0])
        new_offset = offset * multiplier
        new_figure = new_offset + new_interval[0]
        input_vec[i] = new_figure

    return input_vec


def holdout_split(sample_sequences, hold_out_perc=80):
    # This function returns the random holdout split for the given data

    # Ar first you shuffle the data
    random.shuffle(sample_sequences)

    # You compute the length and the size of the split
    length = len(sample_sequences)
    train_length = int(length * hold_out_perc / 100)

    test_set = []
    train_set = []
    n = 0

    # We pick the first train_length lines for the train_set and the resting for the testing
    print 'Preparing hold-out split'
    for i in pbar(sample_sequences):
        if n < train_length:
            train_set.append(i)
        else:
            test_set.append(i)

        n = n + 1

    return train_set, test_set


def split_sets(train_set, test_set):
    # This function returns the two sets: inputs, targets

    X_tr = []
    Y_tr = []
    for t in train_set:
        X_tr.append(t[0])
        Y_tr.append(t[1])

    X_te = []
    Y_te = []
    for t in train_set:
        X_te.append(t[0])
        Y_te.append(t[1])

    return np.array(X_tr), np.array(Y_tr), np.array(X_te), np.array(Y_te)


def main():

    if args.split_file == None:
        ss, countries_number = import_data(args.seq_length,args.target_distance,args.number_of_samples,args.at_least)
        file = open('split.txt', 'wb')
        pickle.dump((ss, countries_number), file)
        file.close()
    else:
        file = open('split.txt', 'rb')
        ss, countries_number = pickle.load(file)
        file.close()
        np.savetxt("foo.csv", ss, delimiter=",")

    t, tt = holdout_split(ss)

    mod = build_model(args.seq_length, countries_number) #######
    data = split_sets(t, tt)
    run_network(mod, data)


def build_model(sequence_size, features=None):

    dropout_perc = 0.2

    model = Sequential()

    model.add(LSTM(100, input_shape=(None, features), return_sequences=True))
    model.add(Dropout(dropout_perc))

    # return sequences must be set to false when it is time to get the final output
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(dropout_perc))

    model.add(Dense(features))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start

    return model


def run_network(model=None, data=None):

    global_start_time = time.time()
    
    epochs = 1
    ratio = 0.5
    path_to_dataset = 'household_power_consumption.txt'

    X_train, y_train, X_test, y_test = data

    print("X Shape")
    print(X_train.shape)

    print("Y Shape")
    print(y_train.shape)
    
    print("altre info")
    print(len(X_train))
    print(len(y_train))
    print(X_train[0].shape)
    print(y_train[0].shape)

    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    print '\nData Loaded. Compiling...\n'
    
    if model is None:
        model = build_model()

    try:
        model.fit(
            X_train, y_train,
            batch_size=1, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:100, 0])
        plt.plot(predicted[:100, 0])
        plt.show()
    except Exception as e:
        print str(e)
    print 'Training duration (s) : ', time.time() - global_start_time
    return model, y_test, predicted


main()



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



^^^^^^^^^^^^
TO LOAD THE MODEL CHECK HERE
