import warnings
from keras.optimizers import adam_v2
import numpy as np
from numpy import asarray
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import glob
import cv2
import os
import os.path

warnings.simplefilter(action='ignore', category=FutureWarning)

## Specifying path to database of train and test.

imagePath1 = 'dataset/aug/test/'
imagePath2 = 'dataset/aug/train/'

## Image Processing
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        validation_split = 0.2
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255
)


traindataset = train_gen.flow_from_directory(
                directory= imagePath1,
                color_mode='grayscale',
                class_mode='categorical',
                batch_size=8,
                shuffle=True

)

testdataset = test_gen.flow_from_directory(
                directory= imagePath2,
                color_mode='grayscale',
                class_mode='categorical',
                batch_size=8,
                shuffle=True
)

print(type(traindataset))
print(type(testdataset))
## Data Prep Additional; changing train and test img/labels into nparray.

dataX = asarray(traindataset)
dataY = asarray(testdataset)

#train_images = np.array([example['image'].numpy()[:,:,0] for example in traindataset])
#train_labels = np.array([example['label'].numpy() for example in traindataset])

#test_images = np.array([example['image'].numpy()[:,:,0] for example in testdataset])
#test_labels = np.array([example['label'].numpy() for example in testdataset])


## Concatonating the data
## dynamiczna generacja obrazków i następnie wrzucenie ich do łączenia
np.concatenate((dataX), axis = None)


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
model.compile(loss= 'mean_squared_error',
              optimizer = adam_v2,
              metrics=['accuracy']
              )


model_fit = model.fit(dataX,
                      dataY,
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