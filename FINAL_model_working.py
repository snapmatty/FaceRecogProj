##File Handling Libs
import os
## Image aug + model libs
from keras.optimizer_v1 import adam
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.losses import mean_squared_error
from keras.metrics import accuracy
import cv2
from cv2 import cv2
## Graphical libs
import matplotlib.pyplot as plt
##Mathematical libs
import numpy as np
import random

## Getting all the files into one list
def getListOfFiles(dirName):
    # creating a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterating over all the entries
    for entry in listOfFile:
        # Creating full path
        fullPath = os.path.join(dirName, entry)  # path for each entry in the list
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


# Create pairs of augmented images (if pair is from the same directory they are set as TRUE (1) if not FALSE (0) 1/0 approach)
def generateAugumentedImagePairs(source_dir):
    print('Creating pairs from:', source_dir)

    listOfFiles = getListOfFiles(source_dir)
    print("Elements of list:", len(listOfFiles))
    listOfImages = []
    for filename in listOfFiles:
        image = normalizeImage(augumentImage(greyImage(readImageFromFile(filename))))
        for x in range(0, 5):
            listOfImages.append([filename, image])

    print("Shuffling the images...")
    random.shuffle(listOfImages)

    listOfCombinedImages = []
    temp = None
    for (filename, image) in listOfImages:
        if temp is None:
            temp = (filename, image)
        else:
            c = (combineImg(temp[1], image), areSameFolder(temp[0], filename))
            listOfCombinedImages.append(c)
            temp = None
    print("Returning list of CombinedImages...")
    print("Number of items in list of CombinedImages: ", len(listOfCombinedImages))
    return listOfCombinedImages

## Checking if images are from same folder path (if yes = true; if not = false)
def areSameFolder(filename1, filename2):
    return bool(filename1.rsplit('/', -1)[3] == filename2.rsplit('/', -1)[3])

## Reading the image from file
def readImageFromFile(filename):
    image = cv2.imread(filename)
    return image

## Combining two images into 1
def combineImg(img1, img2):
    return np.concatenate((img1, img2), axis=1)

## Applying augmentations (noise + rotation)
def augumentImage(img):
    # var = 50
    # dev = var * random.random()
    # noise = np.random.normal(0, dev, img.shape)
    # img = img.astype(np.uint8)
    # img += noise.astype(np.uint8)
    # Generate Gaussian noise
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    # rot = np.rot90(img, k=1)

    np.clip(img_gauss, 0., 255.)
    # print(img.shape)
    # img += rot + gray

    # print(img_gauss.shape)
    return img_gauss

## Applying GreyScale to image
def greyImage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Applying equal size to all images
def normalizeImage(img):
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


#  Data Preparation
source = '/content/dataset'

listOfCombinedImages = generateAugumentedImagePairs(source)

print("Splitting the pairs into train, val and test:")
list_of_pairs_with_vals_train = listOfCombinedImages[:int(0.8 * len(listOfCombinedImages))]
list_of_pairs_with_vals_val = listOfCombinedImages[int(0.15 * len(listOfCombinedImages)):]
list_of_pairs_with_vals_test = listOfCombinedImages[int(0.05 * len(listOfCombinedImages)):]

print("- Train batch:", len(list_of_pairs_with_vals_train))
print("- Test batch:", len(list_of_pairs_with_vals_test))
print("- Validation batch:", len(list_of_pairs_with_vals_val))

print(type(listOfCombinedImages))

## Model & Fitment
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 128, 1)),
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
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])
model.summary()

print('Model is being built!')
model.compile(loss=mean_squared_error,
              optimizer="adam",
              metrics=["accuracy"])
print("Fitting data into model!")


x = []
for item in list_of_pairs_with_vals_train:
  image = item[0]
  x.append(image)
x = np.stack(x, 0)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape)

# get the target training data (is matched or not)
y = []
for item in list_of_pairs_with_vals_train:
  result = item[1]
  y.append(result)
y = np.array(y)

model_fit = model.fit(x,
                      y,
                      epochs=20)

model.save('dataset/')