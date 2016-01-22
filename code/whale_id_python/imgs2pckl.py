__author__ = 'niklas'

import numpy as np
import os
from PIL import Image
import pickle

pathjoin = os.path.join

# define dirnames
data_dir = pathjoin(".." , "..", "224x224")
img_dir = pathjoin(data_dir, "images")
pickle_dir = pathjoin(data_dir, "python_pickle")

# prototyping for when just trying to verify that things work as they should
prototype = False
if prototype:
    img_dir = pathjoin(data_dir, "tiny_sample")

print("Using tiny prototype sample:{}".format(prototype))

# find image paths
print("finding image paths")
whale_picture_names = os.listdir(img_dir)

# filenames are sorted lexicographically, which does not always correspond to numerical ordering of ids
# e.g. after '1' there is '10005'
# b/c whale pictures are linked to classes by IDs we need to keep track of this orderly
# another caveat: there is exactly one ID missing (7489). So we can't just say that #row of matrix = ID of picture
# unless we want to always have to think about that after 7488 it is row of matrix = ID-1 or something
# that seems unreasonable
# so let's save this somewhere
whale_ids = [int(w.split(sep=".")[0].split(sep="_")[1]) for w in whale_picture_names]

# load pictures
print("loading pictures")
all_pictures_matrix = None
n_pictures = len(whale_picture_names)
for picture_index, picture_name in enumerate(whale_picture_names):

    print("Processing picture {} out of {}".format(picture_index, n_pictures))

    # construct path, load picture data
    picture_path = pathjoin(img_dir, picture_name)
    picture_as_array = np.asarray(Image.open(picture_path))

    # for first picture: initialize big matrix with appropriate dimensions
    if picture_index == 0:
        print("initializing big matrix")
        # shape is: #imgs x width x breadth x #channels
        big_matrix_shape = tuple([n_pictures]) + picture_as_array.shape
        all_pictures_matrix = np.empty(big_matrix_shape, dtype=np.int_)

    # save picture into big matrix
    all_pictures_matrix[picture_index:, ...] = picture_as_array

# pickle results
print("pickling big matrix")
if prototype:
    pickle_file_path = pathjoin(pickle_dir, "images_and_ids_prototype.pkl")
else:
    pickle_file_path = pathjoin(pickle_dir, "images_and_ids.pkl")

with open(pickle_file_path, "wb") as pickle_file:
    pickle.dump((all_pictures_matrix, whale_ids), pickle_file)


