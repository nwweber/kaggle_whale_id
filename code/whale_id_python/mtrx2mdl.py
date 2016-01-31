__author__ = 'niklas'

import os
import pickle
import theano
import theano.tensor as T

import lasagne


pathjoin = os.path.join

# goal: run training data through network to train it. then save trained network so it can be used
# to predict the train set

#  === global prototyping switch ===
prototype = True

# === load data ===
print("Loading data from pickle files")
# define dirnames
# also depends on whether you're working locally (on your computer) or on a cluster
local = True
print("Using local computer paths: {}".format(local))
if local:
    data_dir = pathjoin("/", "home", "niklas", "big_datasets", "whales-id", "224x224")
else:
    data_dir = pathjoin("..", "..", "224x224")
img_dir = pathjoin(data_dir, "images")
pickle_dir = pathjoin(data_dir, "python_pickle")

with open(pathjoin(pickle_dir, "train_arrays_prototype={}.pkl".format(prototype)), "rb") as f:
    X_train, Y_train = pickle.load(f)

with open(pathjoin(pickle_dir, "validation_arrays_prototype={}.pkl".format(prototype)), "rb") as f:
    X_validation, Y_validation = pickle.load(f)

# === define network ===
print("defining network")


def create_network():
    pass

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

