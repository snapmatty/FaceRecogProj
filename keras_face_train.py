import warnings
from keras.optimizers import adam_v2
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)

train = ImageDataGenerator()
valid = ImageDataGenerator()
train_dataset = train.flow_from_directory('dataset/aug/train',
                                          target_size=(64, 64),
                                          batch_size=4,
                                          class_mode='binary')

valid_dataset = valid.flow_from_directory('dataset/aug/valid',
                                          target_size=(64, 64),
                                          batch_size=4,
                                          class_mode='binary')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1, activation = 'softmax')
                                    ])
model.add(Dropout(0.2))
model.add(BatchNormalization(2))

model.summary()

model.compile(loss= 'binary_crossentropy',
              optimizer=adam_v2(learning_rate=0.001),
              metrics=['accuracy'])

model_fit = model.fit(training_data = train_dataset,
                      validation_data=valid_dataset,
                      step_per_epoch = 3,
                      epochs = 20,
                      verbose =2
                    )