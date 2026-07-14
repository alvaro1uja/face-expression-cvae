import os
import time
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model

#sampling layer
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        sigma = tf.exp(0.5*logvar)        
        return mean + sigma*epsilon

# encoder
def build_encoder(num_classes, latent_dim):
    #input layers
    img_input = keras.Input(shape=(112, 112, 3))
    label_input = keras.Input(shape=(num_classes,))

    #hidden layers
    #(112,112,3)
    conv1 = keras.layers.Conv2D(30, 3, activation="relu", padding="same")(img_input)
    conv1_2 = keras.layers.Conv2D(30, 3, activation="relu", padding="same")(conv1)
    #(112,112,30)
    pooling1 = keras.layers.MaxPooling2D(2)(conv1_2)
    #(56,56,30)
    conv2 = keras.layers.Conv2D(60, 3, activation="relu", padding="same")(pooling1)
    conv2_2 = keras.layers.Conv2D(60, 3, activation="relu", padding="same")(conv2)
    #(56,56,60)
    pooling2 = keras.layers.MaxPooling2D(2)(conv2_2)
    #(28,28,60)
    conv3 = keras.layers.Conv2D(120, 3, activation="relu", padding="same")(pooling2)
    conv3_2 = keras.layers.Conv2D(120, 3, activation="relu", padding="same")(conv3)
    #(28,28,120)
    pooling3 = keras.layers.MaxPooling2D(2)(conv3_2)
    #(14,14,120)

    flat = keras.layers.Flatten()(pooling3)
    #(23520)
    concat = keras.layers.Concatenate()([flat, label_input])
    #(23520 + 4)

    dense1 = keras.layers.Dense(512, activation="relu")(concat)
    #(512)
    dense2 = keras.layers.Dense(256, activation="relu")(dense1)
    #(256)

    #output layers
    mean = keras.layers.Dense(latent_dim)(dense2)
    logvar = keras.layers.Dense(latent_dim)(dense2)
    z = Sampling()([mean, logvar])
    
    return keras.Model([img_input, label_input], [mean, logvar, z])


# decoder
def build_decoder(num_classes, latent_dim):
    #input layers
    z_input = keras.Input(shape=(latent_dim,))
    label_input = keras.Input(shape=(num_classes,))

    #hidden layers
    concat = keras.layers.Concatenate()([z_input, label_input])
    #(latent_dim + num_classes)
    
    dense1 = keras.layers.Dense(256, activation="relu")(concat)
    #(256)
    dense2 = keras.layers.Dense(512, activation="relu")(dense1)
    #(512)

    dense5 = keras.layers.Dense(14*14*120, activation="relu")(dense2)
    reshape1 = keras.layers.Reshape((14,14,120))(dense5)
    #(14,14,120)
    conv_transpose1 = keras.layers.Conv2DTranspose(120, 3, strides=2, padding="same", activation="relu")(reshape1)
    conv1 = keras.layers.Conv2D(120, 3, activation="relu", padding="same")(conv_transpose1)
    #(28,28,120)
    conv_transpose2 = keras.layers.Conv2DTranspose(60, 3, strides=2, padding="same", activation="relu")(conv1)
    conv2 = keras.layers.Conv2D(60, 3, activation="relu", padding="same")(conv_transpose2)
    #(56,56,60)
    conv_transpose3 = keras.layers.Conv2DTranspose(30, 3, strides=2, padding="same", activation="relu")(conv2)
    conv3 = keras.layers.Conv2D(30, 3, activation="relu", padding="same")(conv_transpose3)
    #(112,112,30)

    #output layers
    output = keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same")(conv3)
    #(112,112,3)

    return keras.Model([z_input, label_input], output)

#loss function
def vae_loss(x, xp, mean, logvar):
    beta = 1.0
    #reconstruction loss
    recon = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(x -xp), axis=[1, 2, 3]))

    #KL loss
    kl = -0.5*tf.reduce_mean(tf.reduce_sum(1 + logvar -tf.square(mean) -tf.exp(logvar), axis=1))

    loss = recon + beta*kl

    return loss, recon, kl


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_tr = keras.metrics.Mean(name="loss")
        self.recon_tr = keras.metrics.Mean(name="recon")
        self.kl_tr = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tr, self.recon_tr, self.kl_tr]

    def call(self, inputs):
        x, y = inputs
        mean, logvar, z = self.encoder([x, y])
        xp = self.decoder([z, y])
        return xp

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            mean, logvar, z = self.encoder([x, y])
            xp = self.decoder([z, y])

            loss, recon, kl = vae_loss(x, xp, mean, logvar)

        weights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        self.loss_tr.update_state(loss)
        self.recon_tr.update_state(recon)
        self.kl_tr.update_state(kl)

        return {"loss": self.loss_tr.result(), "recon": self.recon_tr.result(), "kl": self.kl_tr.result()}

    def test_step(self, data):
        x, y = data

        mean, logvar, z = self.encoder([x, y])
        xp = self.decoder([z, y])

        loss, recon, kl = vae_loss(x, xp, mean, logvar)

        self.loss_tr.update_state(loss)
        self.recon_tr.update_state(recon)
        self.kl_tr.update_state(kl)

        return {"loss": self.loss_tr.result(), "recon": self.recon_tr.result(), "kl": self.kl_tr.result()}
