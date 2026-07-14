import os
import time
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model

class DataLoader(keras.utils.Sequence):
    """A simple data loader for Assignment 1.

    Note: We advice against making changes to the data loader!
    """

    def __init__(self,
                 data_path,
                 class_map,
                 batch_size=32,
                 cache=True,
                 random_state=None,
                 dtype=np.uint8,
                 ):

        self.data_path = data_path
        self.class_map = class_map
        self.batch_size = max(1, int(batch_size))
        self.cache = bool(cache)
        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self.dtype = dtype

        if self.data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(self.data_path):
            raise ValueError('The data path is incorrectly defined.')

        if not isinstance(self.class_map, dict):
            raise ValueError('The folder map is not a dictionary.')

        # Read the files in all subfolders
        self._file_idx = 0
        self._images = []
        self._labels = []
        for folder in self.class_map:
            path = os.path.join(self.data_path, folder)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                self._images.append(file_path)
                self._labels.append(folder)

        self._image_cache = dict()

        self.on_epoch_end()

    def __len__(self):
        """Get the number of mini-batches per epoch."""
        return int(len(self._images) / self.batch_size)

    def __getitem__(self, index):
        """Get one batch of data."""
        # Generate indices of the batch
        indices = self._indices[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Find the next set of file indices
        minibatch_files = [self._images[k] for k in indices]
        minibatch_labels = [self.class_map[self._labels[k]] for k in indices]

        # Load up the corresponding minibatch
        minibatch = self.__load_minibatch(minibatch_files)

        X = np.asarray(minibatch, dtype=np.float32) / 255.0
        y = np.asarray(minibatch_labels, dtype=np.float32)
        
        return X, y
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        self._indices = np.arange(len(self._images))
        self.random_state.shuffle(self._indices)

    def __load_image(self, file):
        """Load a single image from file."""
        im = Image.open(file)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = np.asarray(im, dtype=self.dtype)

        return im

    def __load_minibatch(self, minibatch_files):
        """Load the next minibatch of samples."""

        try:
            assert self.batch_size == len(minibatch_files)
        except AssertionError:
            print(self.batch_size)
            print(len(minibatch_files))

        minibatch = [None] * self.batch_size
        for i, file in enumerate(minibatch_files):
            if self.cache:
                if file in self._image_cache:
                    im = self._image_cache[file]
                else:
                    im = self.__load_image(file)
                    self._image_cache[file] = im
            else:
                im = self.__load_image(file)

            minibatch[i] = im

        return minibatch