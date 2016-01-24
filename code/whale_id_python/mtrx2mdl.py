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

# === split into train / validation set
n_pictures = pictures.shape[0]

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

