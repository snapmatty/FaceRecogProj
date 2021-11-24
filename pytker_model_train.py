import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

directory = "dataset/"
actor_class = ["Brad Pitt", "Christian Bale", "Johnny Depp", "Keanu Reeves", "Steve Jobs"]

for category in actor_class:
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_arr, cmap="gray")
            plt.show()
            break
        break







