__author__ = 'niklas'

import numpy as np
import os
from PIL import Image
import pickle
import glob
import pandas as pd

pathjoin = os.path.join

# goal: run training data through network to train it. then save trained network so it can be used
# to predict the train set


def extract_id(filename):
    """
    take a filename of the form 'w_ID.[pikl/jpg/...]', return ID as an integer
    :param filename:
    :return:
    """
    return int(filename.split(sep=".")[0].split(sep="_")[1])


#  === global prototyping switch ===
prototype = True

# === load data ===
print("Loading data from pickle files")
# define dirnames
# also depends on whether you're working locally (on your computer) or on a cluster
local = False
print("Using local computer paths: {}".format(local))
if local:
    data_dir = pathjoin("/", "home", "niklas", "big_datasets", "whales-id", "224x224")
else:
    data_dir = pathjoin("..", "..", "224x224")
img_dir = pathjoin(data_dir, "images")
pickle_dir = pathjoin(data_dir, "python_pickle")

pickle_file_paths = glob.glob(pathjoin(pickle_dir, "w_*.pkl"))

# a list of whale picture ids matching the order in which the files will be read in
whale_picture_ids = [extract_id(os.path.basename(pfile_path)) for pfile_path in pickle_file_paths]

n_images = len(pickle_file_paths)
images_list = []
for pfile_index, pfile_path in enumerate(pickle_file_paths):
    print("loading image {} out of {}".format(pfile_index+1, n_images))
    with open(pfile_path, "rb") as f:
        images_list.append(pickle.load(f))

# === split into train / validation set
# only keep images with known class
# randomly split into train/validation
# pickle completed train/validation
class_data_csv_path = pathjoin(data_dir, "..", "train.csv")


# === define network ===
print("defining network")
network = create_network()

# === mini-batch train network ===
n_epochs = 500
validation_errors = []
for epoch_index in range(n_epochs):
    for minibatch in iterate_minibatches():
        train_network(network, minibatch)
    validation_errors.append(validate_network(network, X_valid, Y_valid))
    # output / log validation error
    print("Epoch {} finished. Validation error: {}".format(epoch_index, validation_errors[epoch_index]))
    # from time to time save intermittent network weights
    if epoch % 20 == 0:
        save_network(network)

# === save final network weights ===
print("Training finished, saving final network")
save_network(network)

# === display/save final stats ===

