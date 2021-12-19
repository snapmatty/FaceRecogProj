##File Handling Libs
import os
import csv
## Image aug + model libs
# import tensorflow as tf
import warnings
from keras.optimizer_v1 import adam
import numpy as np
from numpy import asarray
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.losses import mean_squared_error
from keras.metrics import accuracy

## Graphical libs
import matplotlib.pyplot as plt
##Mathematical libs
import numpy as np
import random


def getListOfFiles(dirName):
    # creating a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterating over all the entries
    for entry in listOfFile:
        # Creating full path
        fullPath = os.path.join(dirName, entry) # path for each entry in the list
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


# Create CSV files with pairs (if pair is from the same directory they are set as TRUE (1) if not FALSE (0) 1/0 approach
def create_csv_files(source_dir, filename_trainVal, filename_test):
    print('Creating pairs from:', source_dir)

    listOfFiles = getListOfFiles(source_dir)

    #list_of_valid = random.sample(listOfFiles, 10)
    list_of_test = random.sample(listOfFiles, 10)
    list_of_train = [file for file in listOfFiles if not file in list_of_test]
    # list_of_train, list_of_test = train_test_split(listOfFiles, test_size=0.3, random_state=111)

    with open(filename_trainVal, 'w', newline='') as f:
       write = csv.writer(f)
       # write.writerow(["Pic1", "Pic2", "Are_The_Same_Person"])
       # print(list_of_train)
       list_of_pairs_with_vals = []


       for pic1 in list_of_train:
           for pic2 in list_of_train:
                        if pic1 == pic2:
                            continue
                        val = bool(pic1.rsplit('\\', -1)[1] == pic2.rsplit('\\', -1)[1]) ##splitting so that we only get the acutal name of the actor from the 'dataset / ACTORNAME / IMG' path
                        list_of_pairs_with_vals.append((pic1, pic2, val)) ## appending the two images obtained and their value to the created list
       write.writerows(list_of_pairs_with_vals)  ## they are written into a row

## the same procedure as above is going to happen for the TEST samples
    with open(filename_test, 'w', newline='') as f:
       write = csv.writer(f)
       list_of_pairs_with_vals = []
       for pic1 in list_of_test:
            for pic2 in list_of_test:
                val = bool(pic1.rsplit('\\', -1)[1] == pic2.rsplit('\\', -1)[1])
                list_of_pairs_with_vals.append((pic1, pic2, val))
       write.writerows(list_of_pairs_with_vals)
    return



#reading from the CSV file
def read_from_csv(filename):
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)

        list_of_pairs_with_vals = []

        for row in reader:
            list_of_pairs_with_vals.append(row)

        return list_of_pairs_with_vals



def data_augmentation_function(img, label):
    if label < 1000:
        np.rot15(img, k=label)
        np.random.normal(0,1,100)

        return(img, label)

    raise ValueError


#  Data Preparation
source = 'dataset\\'
csv_file_train = 'pairs_with_value_train.csv'
csv_file_test = 'pairs_with_value_test.csv'
#csv_file_valid = 'pair_with_value_valid.csv'

create_csv_files(source, csv_file_train, csv_file_test)

list_of_pairs_with_vals_test = read_from_csv(csv_file_test)
list_of_pairs_with_vals_train_val = read_from_csv(csv_file_train)

random.shuffle(list_of_pairs_with_vals_train_val)

list_of_pairs_with_vals_train = list_of_pairs_with_vals_train_val[:int(0.8 * len(list_of_pairs_with_vals_train_val))]
list_of_pairs_with_vals_val   = list_of_pairs_with_vals_train_val[int(0.8 * len(list_of_pairs_with_vals_train_val)):]
#the operations above just create the split into 90 and 10% batches of Valid and Train, can also be done once more to split into one more batch of testing batch
del list_of_pairs_with_vals_train_val  #releasing the memory


print(list_of_pairs_with_vals_val)

data_augmentation_function(list_of_pairs_with_vals_train[::1], label = list_of_pairs_with_vals_train[:1])


#  Model Prep




# Model Train
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 64, 1)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    BatchNormalization(2),

                                    #
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    Dropout(0.2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    BatchNormalization(2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                                    ])
model.summary()




print('Model is being built!')
model.compile(loss= mean_squared_error,
              optimizer = adam,
              metrics=[accuracy]
              )


model_fit = model.fit(list_of_pairs_with_vals_train,
                      epochs = 20,
                      callbacks = [
                          tf.keras.callbacks.EarlyStopping(
                              monitor = 'val_loss',
                              patience = 5,
                              restore_best_weights = True
                          )

                      ]
                   )

model.save('dataset/')


