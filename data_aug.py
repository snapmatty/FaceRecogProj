import cv2
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import random

# images = dir('dataset/')
# noise_gauss2 = np.random.normal(0,1,100)
# noise_gauss = random_noise(images, mode='gaussian', seed = None, clip = True)


cv_img = []
for img in glob.glob("dataset/*.jpg"):
    n = cv2.imread(img)
    cv_img.append(n)

array_of_img = np.array(cv_img)

def add_gauss_noise(array_of_img):
    '''Add random noise to an image'''
    var = 50
    dev = var * random.random()
    noise = np.random.normal(0, dev, array_of_img.shape)
    array_of_img += noise
    np.clip(array_of_img, 0., 255.)
    return array_of_img


generating_data = ImageDataGenerator(
    rotation_range=5,
    preprocessing_function = add_gauss_noise(array_of_img)
)

i = 0
for batch in generating_data.flow_from_directory(
        directory='dataset/',
        color_mode='rgb',
        batch_size=50,
        shuffle=True,
        target_size=(256, 256),
        save_to_dir='testofimages',
        save_prefix='aug',
        save_format='jpg'):
    i += 1
    if i > 300:
        break
