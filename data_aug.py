import cv2
from keras_preprocessing.image import ImageDataGenerator, io
import numpy as np
from matplotlib import pyplot
from PIL import Image
from skimage import io
from skimage.filters import gaussian


generating_data = ImageDataGenerator(
        rotation_range = 5,
        horizontal_flip = True,
)

i = 0
for batch in generating_data.flow_from_directory(
                                directory='dataset/',
                                color_mode='rgb',
                                batch_size = 2,
                                interpolation= "bilinear",
                                shuffle = True,
                                target_size= (256,256),
                                save_to_dir = 'testofimages',
                                save_prefix = 'aug',
                                save_format = 'jpg'):
    i +=1
    if i>20:
        break

