__author__ = 'niklas'

import math
import os
import pickle
import glob

import numpy as np
import pandas as pd
from PIL import Image

pathjoin = os.path.join

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

# prototyping for when just trying to verify that things work as they should
prototype = True
if prototype:
    img_dir = pathjoin(data_dir, "tiny_sample")

print("Using tiny prototype sample: {}".format(prototype))

# find image paths
print("finding image paths")
whale_picture_names = os.listdir(img_dir)

# load pictures
print("loading pictures")
# all_pictures_matrix = None
n_pictures = len(whale_picture_names)
for picture_index, picture_name in enumerate(whale_picture_names):

    print("Processing picture {} out of {}".format(picture_index, n_pictures))

    # construct path, load picture data
    picture_path = pathjoin(img_dir, picture_name)
    picture_as_array = np.asarray(Image.open(picture_path))

    # pickle image
    pickle_name = picture_name.split(".")[0] + ".pkl"
    pickle_file_path = pathjoin(pickle_dir, pickle_name)
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(picture_as_array, pickle_file)

def extract_image_id(filename):
    """
    take a filename of the form 'w_ID.[pikl/jpg/...]', return ID as an integer
    :param filename:
    :return:
    """
    return int(filename.split(sep=".")[0].split(sep="_")[1])


def tuple_list_to_arrays(tuple_list):
    """
    transform list of ( (width x breadth x #channels), class ) tuples into
    X/Y arrays, with X having shape (#imgs x #channels x w x h) and Y
    having shape (#imgs,)
    :param tuple_list:
    :return:
    """
    n_images = len(tuple_list)
    head_image = tuple_list[0][0]
    w, h, ch = head_image.shape
    X = np.empty((n_images, ch, w, h), dtype=head_image.dtype)
    Y = np.empty((n_images,))
    for i, (image, class_label) in enumerate(tuple_list):
        image_swapped = image.swapaxes(0, 2)
        X[i, :, :, :] = image_swapped
        Y[i] = class_label
    return X, Y

# === load data ===
print("Loading data from pickle files")
print("Using local computer paths: {}".format(local))
if local:
    data_dir = pathjoin("/", "home", "niklas", "big_datasets", "whales-id", "224x224")
else:
    data_dir = pathjoin("..", "..", "224x224")
img_dir = pathjoin(data_dir, "images")
pickle_dir = pathjoin(data_dir, "python_pickle")

pickle_file_paths = glob.glob(pathjoin(pickle_dir, "w_*.pkl"))

if prototype:
    pickle_file_paths = pickle_file_paths[:10]

# a list of whale picture ids matching the order in which the files will be read in
whale_image_ids = [extract_image_id(os.path.basename(pfile_path)) for pfile_path in pickle_file_paths]

n_images_in_dataset = len(pickle_file_paths)
images_list = []
for pfile_index, pfile_path in enumerate(pickle_file_paths):
    print("loading image {} out of {}".format(pfile_index+1, n_images_in_dataset))
    with open(pfile_path, "rb") as f:
        images_list.append(pickle.load(f))

# === split into train / validation set
# only keep images with known class
# randomly split into train/validation
# pickle completed train/validation
class_data_csv_path = pathjoin(data_dir, "..", "train.csv")
class_data = pd.read_csv(class_data_csv_path)

image_class_tuples = []
for index, row_data in class_data.iterrows():
    image_id = extract_image_id(row_data["Image"])
    whale_id = int(row_data["whaleID"].split("_")[1])
    # image id might not be present in list of iIDs if only using subset of images, e.g. for prototyping
    try:
        image_index = whale_image_ids.index(image_id)
        image_class_tuples.append((images_list[image_index], whale_id))
    except ValueError:
        print("image id {} not found in list, skipping".format(image_id))
        continue

n_labelled_images = len(image_class_tuples)
train_fraction = 0.8
n_train_images = math.floor(train_fraction * n_labelled_images)
n_validation_images = n_labelled_images - n_train_images

# note: not a randomized split. then again, images are probably in a somewhat-random order anyway
# if randomness desired: randomize indices into this list, then make new list based on these indices
# then take slices of this list
train_tuples = image_class_tuples[:n_train_images]
test_tuples = image_class_tuples[n_train_images:]

X_train, Y_train = tuple_list_to_arrays(train_tuples)
X_validation, Y_validation = tuple_list_to_arrays(test_tuples)

with open(pathjoin(pickle_dir, "train_arrays_prototype={}.pkl".format(prototype)), "wb") as f:
    pickle.dump((X_train, Y_train), f, protocol=4)

with open(pathjoin(pickle_dir, "validation_arrays_prototype={}.pkl".format(prototype)), "wb") as f:
    pickle.dump((X_validation, Y_validation), f, protocol=4)

