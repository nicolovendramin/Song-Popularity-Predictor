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

def sample(samp_num, min_date_, max_offset_, data, top_required_position, input_length, countries, worst_position, country_to_id, target_distances, scale):
    selected_points = [[] for i in range(0, len(target_distances))]
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
                inputs.append(input_array)

            # we fetch the target as one of the inputs

            # We append the pair (list of inputs [position in country array for the considered time window]
            # -> target) to the list of selected points
            inpt = inputs[0]
            for i in range(1, len(inputs)):
                inpt = np.vstack((inpt, inputs[i]))
            
            index_of_the_list = 0
            for target_distance in target_distances:
                target = np.full(len(countries), float(-1))

                day =  starting_date + dt.timedelta(input_length + target_distance - 1)

                daily = data['Date'] == str(day)
                author = data['Artist'] == song[1]
                track_name = data['Track Name'] == song[0]

                result = data[daily & author & track_name][['Position', 'Region']]
                for index, row in result.iterrows():
                    target[country_to_id[row['Region']]] = row['Position']

                selected_points[index_of_the_list].append((inpt, target))
                index_of_the_list += 1

        # In case the draw was not ok, we add one more to be done and we skip this iteration
        else:
            print("daily songs number too low:", daily_num)
            samp_num = samp_num + 1

    print(len(selected_points))
    return selected_points

class DataPreparation:
      
    @staticmethod    
    def import_data(input_length, target_distance, number_of_samples, top_required_position=200, dump_folder="data/sequences", processors=4, scale = True):
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
        max_offset = time_delta.days - input_length - max(target_distance) - 1

        # Starts selecting the samples iterating of a cycle of size number_of_samples
        pool = multiprocessing.Pool()
        blocks = []
        min_date_ = dt.datetime.strptime(min_date, '%Y-%m-%d').date()
        max_offset_ = (time_delta.days - input_length - max(target_distance) - 1) / processors 
        sample_num = int(number_of_samples / processors) + 1
        for i in range(0, processors):
            print("Running instance {}".format(i+1))
            blocks.append(pool.apply_async(sample, (sample_num, min_date_, max_offset_, data, top_required_position, input_length, countries, worst_position,  country_to_id, target_distance, scale),))
            min_date_ = min_date_ + dt.timedelta(max_offset_)

        selected_points = [[] for i in range(0, len(target_distance))]
        # Fetching of the results from the workers who run the tasks
        for j in range(0, processors):
            sp = blocks[j].get()
            print(len(sp))
            for sub_ in range(0, len(selected_points)):
                selected_points[sub_] += sp[sub_]

        descriptions = ""

        # selected_points = sample(number_of_samples, min_date, max_offset)
        for i in range(0, len(target_distance)):
            description = str(input_length) + "_" + str(target_distance[i]) + "_" + str(number_of_samples) + "_" + str(top_required_position)
            descriptions += description + "\n"
            if dump_folder != None:
                file = open(dump_folder + "/" + description + ".txt", 'wb')
                pickle.dump((selected_points[i], len(countries)), file)
                file.close()
                print("Dump file produced.")

        return selected_points, len(countries), descriptions


    def put_as_single_series(set_):
        result = [[] for i in list(set_[0][0])]
        a = len(result)
        print(a)
        for sample in set_:
            for time_moment in sample:
                part_ = [[] for i in list(set_[0][0])]
                print(part_)
                for i in range(0, a):
                    part_[i].append(time_moment[i])
            for i in range(0, a):
                result[i].append(np.asarray(part_[i]))
                print(result[i])

        return np.asarray(result)

    def put_as_single_series_(set_):
        result = [[] for i in list(set_[0])]
        a = len(result)
        print(a)
        for sample in set_:
            for i in range(0, a):
                    result[i].append(sample[i])

        return np.asarray(result)






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













