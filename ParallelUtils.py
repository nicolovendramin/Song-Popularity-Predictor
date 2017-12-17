import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import random
import time
import argparse
import pickle
from progressbar import ProgressBar
import multiprocessing
import math as m
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

pbar = ProgressBar()

parser = argparse.ArgumentParser()
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--target_distance', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=6)
parser.add_argument('--number_of_samples', type=int, default=100)
parser.add_argument('--at_least', type=int, default=100)
args = parser.parse_args()

def sample(samp_num, min_date_, max_offset_, data, top_required_position, input_length, countries, worst_position, country_to_id, target_distance, scale):
    selected_points = []
    pbar = ProgressBar()
    for n in pbar(range(0, samp_num)):
        # Randomly selects the day to start the sequence respecting the boundary of having a complete sequence
        starting_date = min_date_ + dt.timedelta(np.random.randint(0, max_offset_))

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
                inputs.append(DataPreparation.rescaling(input_array, (-1, 200), (-1,1), to_do = scale))

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


            selected_points.append((inpt, DataPreparation.rescaling(target, (-1, 200), (-1,1), to_do=scale)))

        # In case the draw was not ok, we add one more to be done and we skip this iteration
        else:
            print("daily songs number too low:", daily_num)
            samp_num = samp_num + 1

    return selected_points

class DataPreparation:
      
    @staticmethod    
    def import_data(input_length, target_distance, number_of_samples, top_required_position=200, dump_folder="data/sequences", processors=4, scale = True):
        # This function returns a dataset including a number of evolunetions for a song in the national charts among all the
        # countries during a number (input_length) of days, and the ranking for the same song in all the charts some
        # (target_distance) days after. The last parameter is the number of elements in the set. Default is extracting all
        # the possible series.

        # writing the description of the data
        description = str(input_length) + "_" + str(target_distance) + "_" + str(number_of_samples) + "_" + str(top_required_position)  

        if dump_folder != None:
            try:
                file = open(dump_folder + "/" + description + ".txt", 'rb')
                selected_points, countries_number = pickle.load(file)
                file.close()
                print("Loading files from previous run.")
                return selected_points, countries_number, description
            except Exception as e:
                print("No previous run to be resumed.") 

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
        pool = multiprocessing.Pool()
        blocks = []
        min_date_ = dt.datetime.strptime(min_date, '%Y-%m-%d').date()
        max_offset_ = (time_delta.days - input_length - target_distance - 1) / processors 
        sample_num = int(number_of_samples / processors) + 1
        for i in range(0, processors):
            print("Running instance {}".format(i+1))
            blocks.append(pool.apply_async(sample, (sample_num, min_date_, max_offset_, data, top_required_position, input_length, countries, worst_position,  country_to_id, target_distance, scale),))
            min_date_ = min_date_ + dt.timedelta(max_offset_)

        selected_points = []
        # Fetching of the results from the workers who run the tasks
        for j in range(0, processors):
            selected_points += blocks[j].get()

        # selected_points = sample(number_of_samples, min_date, max_offset)

        if dump_folder != None:
            file = open(dump_folder + "/" + description + ".txt", 'wb')
            pickle.dump((selected_points, len(countries)), file)
            file.close()
            print("Dump file produced.")

        return selected_points, len(countries), description


    @staticmethod
    def rescaling(input_vec, old_interval, new_interval, to_do = True):
    # rescales the input vector in the new interval
        if to_do:
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


    @staticmethod    
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
        print('Preparing hold-out split')
        pbar = ProgressBar()
        for i in pbar(sample_sequences):
            if n < train_length:
                train_set.append(i)
            else:
                test_set.append(i)

            n = n + 1

        return train_set, test_set


    @staticmethod
    def separe_labels(train_set, test_set, dump_file=None):
        # This function returns the two sets: inputs, targets

        if dump_file != None:
            try:
                file = open(dump_file, 'rb')
                X_train, Y_train, X_test, Y_test = pickle.load(file)
                file.close()
                print("Loading files from previous run.")
                return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
            except Exception as e:
                print("No previous run to be resumed.")

        X_train = []
        Y_train = []
        for sequence, label in train_set:
            X_train.append(sequence)
            Y_train.append(label)

        X_test = []
        Y_test = []
        for sequence, label in test_set:
            X_test.append(sequence)
            Y_test.append(label)

        if dump_file != None:
            file = open(dump_file, 'wb')
            pickle.dump((X_train, Y_train, X_test, Y_test), file)
            file.close()
            print("Dump file produced.")

        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


    @staticmethod
    def import_train_test(input_length, target_distance,
     number_of_samples, top_required_position=200, sequence_dumps="data/sequences",
      hold_out_perc=80, splits_dumps="data/splits", processors=4, scale=True):
        
        sequences, country_num, description = DataPreparation.import_data(input_length, target_distance, number_of_samples, top_required_position, sequence_dumps, processors=processors,scale=scale)
        train_set, test_set = DataPreparation.holdout_split(sequences, hold_out_perc)
        X_train, Y_train, X_test, Y_test = DataPreparation.separe_labels(train_set, test_set, splits_dumps + "/" + description + ".txt")

        return X_train, Y_train, X_test, Y_test, description


class BaselineComputation:

    @staticmethod 
    def mae(X, Y):
        sum_ = 0
        len_ = 0
        for pair in zip(X, Y):
            sum_ += m.fabs(pair[0] - pair[1])
            len_ += 1

        return sum_ / len_ 


    @staticmethod
    def constant(X_train, Y_train, X_test, Y_test):

        def predict(vector_list):
            return vector_list[-1]

        size_test = len(Y_test)
        tmp = 0
        predictions = []
        for sample in range(0, size_test):
            vector_ = predict(X_test[sample])
            predictions.append(list(vector_))
            tmp += BaselineComputation.mae(vector_, Y_test[sample].tolist())

        bas = tmp / size_test

        acc = BaselineComputation.accuracy_computer(predictions, X_test, Y_test, "constant")

        return bas, acc


    @staticmethod
    def average(X_train, Y_train, X_test, Y_test):

        def predict(vector_list):
            first_vec = list(vector_list[0])
            len_ = 1

            for vector in vector_list[1:]:
                first_vec += vector
                len_ += 1

            return first_vec / len_

        size_test = len(Y_test)
        tmp = 0
        predictions = []
        for sample in range(0, size_test):
            vector_ = predict(X_test[sample])
            predictions.append(list(vector_))
            tmp += BaselineComputation.mae(vector_, Y_test[sample].tolist())

        bas = tmp / size_test

        acc = BaselineComputation.accuracy_computer(predictions, X_test, Y_test, "average")

        return bas, acc


    @staticmethod
    def random_vector(X_train, Y_train, X_test, Y_test):

        def predict(vector_list):
            vector_ = list(vector_list[0])

            for position in range(0, len(vector_)): 
                vector_[position] = np.random.randint(-1, 200)

            return vector_

        size_test = len(Y_test)
        tmp = 0
        predictions = []
        for sample in range(0, size_test):
            vector_ = predict(X_test[sample])
            predictions.append(list(vector_))
            tmp += BaselineComputation.mae(vector_, Y_test[sample].tolist())

        bas = tmp / size_test

        acc = BaselineComputation.accuracy_computer(predictions, X_test, Y_test, "random vector")

        return bas, acc


    @staticmethod
    def random_increase(X_train, Y_train, X_test, Y_test):

        def predict(vector_list):
            vector = vector_list[-1]
            vector_ = list(vector)

            for position in range(0, len(vector_)): 
                abs_factor = np.random.randn() + 2
                x = [-1, 1]
                np.random.shuffle(x)
                factor = abs_factor * x[0]
                vector_[position] = max(min(vector[position] + factor, 200), -1)

            return vector_

        size_test = len(Y_test)
        tmp = 0
        predictions = []
        for sample in range(0, size_test):
            vector_ = predict(X_test[sample])
            predictions.append(list(vector_))
            tmp += BaselineComputation.mae(vector_, Y_test[sample].tolist())

        bas = tmp / size_test

        acc = BaselineComputation.accuracy_computer(predictions, X_test, Y_test, "random increase")

        return bas, acc


    def linear_regression(X_train, Y_train, X_test, Y_test):
        reg = linear_model.LinearRegression()
        reg.fit([np.ravel(sample).tolist() for sample in X_train], Y_train)
        predictions = reg.predict([np.ravel(sample).tolist() for sample in X_test])
        predictions = [[max(min(prediction, 200), -1) for prediction in vector ]for vector in predictions]

        acc =  BaselineComputation.accuracy_computer(predictions, X_test, Y_test, "linear_regression")

        return mean_absolute_error(Y_test, predictions,multioutput='uniform_average'), acc


    def accuracy_computer(predictions, X_test, Y_test, model):
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(0, len(predictions)):
            diff_pred = predictions[i] - X_test[i][-1]
            diff_true = Y_test[i] - X_test[i][-1]
            for t in range(0, len(predictions[i])):
                if X_test[i][-1][t] > 0 and X_test[i][-1][t] < 199:
                    if diff_true[t] >= 0 and diff_pred[t] >= 0:
                        tp += 1
                    elif diff_true[t] >= 0 and diff_pred[t] < 0:
                        fp += 1
                    elif diff_true[t] < 0 and diff_pred[t] >= 0:
                        fn += 1
                    else:
                        tn += 1

        acc = float(tp+tn) / float(fp+fn+tp+tn)

        return acc


    def compute_baselines(X_train, Y_train, X_test, Y_test, baselines=None):
        available_baselines = {BaselineComputation.average : "average", BaselineComputation.random_vector : "random vector",
     BaselineComputation.random_increase : "random increase", BaselineComputation.constant : "constant",
      BaselineComputation.linear_regression : "linear regression"}

        if baselines == None:
            baselines = available_baselines.keys()

        baseline_names = []
        results = []
        accuracy = []

        for baseline in baselines:
            baseline_names.append(available_baselines[baseline])
            mae, acc = baseline(X_train, Y_train, X_test, Y_test)
            results.append(mae)
            accuracy.append(acc)

        return results, baseline_names, accuracy


class MailServices:

    def send_mail(messages, img_paths, gmailUser='sendmailfrompython@gmail.com', gmailPassword='SendMyMail_', recipient='nicolo.vendramin@gmail.com'):
        import smtplib

        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart()
        msg['From'] = gmailUser
        msg['To'] = recipient
        msg['Subject'] = "Results"
        for message in messages:
            msg.attach(MIMEText(message))

        for path in img_paths:
            file = open(path, 'rb')
            img = MIMEImage(file.read())
            msg.attach(img)

        mailServer = smtplib.SMTP('smtp.gmail.com', 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(gmailUser, gmailPassword)
        mailServer.sendmail(gmailUser, recipient, msg.as_string())
        mailServer.close()













