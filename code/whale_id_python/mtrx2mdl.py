__author__ = 'niklas'

import numpy as np
import os
from PIL import Image
import pickle

pathjoin = os.path.join

# goal: run training data through network to train it. then save trained network so it can be used
# to predict the train set

#  === global prototyping switch ===
prototype = True

# === load data ===
print("Loading data from pickle file")
# define dirnames
data_dir = pathjoin(".." , "..", "224x224")
img_dir = pathjoin(data_dir, "images")
pickle_dir = pathjoin(data_dir, "python_pickle")
if prototype:
    pickle_file_path = pathjoin(pickle_dir, "images_and_ids_prototype.pkl")
else:
    pickle_file_path = pathjoin(pickle_dir, "images_and_ids.pkl")

with open(pickle_file_path, "rb") as pickle_file:
    pictures, ids = pickle.load(pickle_file)

# === define network ===


# === mini-batch train network ===
    # output / log validation error
    # from time to time save intermittent network weights


# === save final network weights ===


# === display/save final stats ===
