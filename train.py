from dataloader import DataLoader
from model import build_encoder, build_decoder, VAE

import os
import time
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.utils import plot_model

def set_seed(seed):
    # seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_plot_settings():
    #for plots
    PLOT_SETTINGS = {"text.usetex": False,
                    "font.family": "serif",
                    "figure.figsize": (8.0, 6.0),
                    "font.size": 16,
                    "axes.labelsize": 16,
                    "legend.fontsize": 14,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "axes.titlesize": 24,
                    "lines.linewidth": 2.0,
                    }
    plt.rcParams.update(PLOT_SETTINGS)

def set_gpu():
    # Check GPU availability (it will be very slow without a GPU...)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print()
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU(s) available. Training will be lightning fast!")
    else:
        print("No GPU(s) available. Training will be very slow ...")# For pretty plots...

def create_dataloaders(data_dir, class_map, batch_size, cache):
    print(f"Loading data from {data_dir}")

    # Create the data loaders
    train_ds = DataLoader(os.path.join(data_dir, "train/"),
                        class_map=class_map,
                        batch_size=batch_size,
                        cache=cache,
                        )
    val_ds = DataLoader(os.path.join(data_dir, "val/"),
                        class_map=class_map,
                        batch_size=batch_size,
                        cache=cache,
                        )
    test_ds = DataLoader(os.path.join(data_dir, "test/"),
                        class_map=class_map,
                        batch_size=batch_size,
                        cache=cache,
                        )

    return train_ds, val_ds, test_ds

def plot_all_loss(history):
    #plot losses
    #total loss
    plt.figure()

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")

    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.title("Total loss")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/loss_total.png")
    plt.close()

    # reconstruction loss
    plt.figure()

    plt.plot(history.history["recon"], label="train")
    plt.plot(history.history["val_recon"], label="val")

    plt.xlabel("epoch")
    plt.ylabel("reconstruction loss")
    plt.title("Reconstruction loss")
    plt.legend()

    plt.savefig("outputs/loss_recon.png")
    plt.close()

    #KL loss
    plt.figure()

    plt.plot(history.history["kl"], label="train")
    plt.plot(history.history["val_kl"], label="val")

    plt.xlabel("epoch")
    plt.ylabel("kl loss")
    plt.title("KL loss")
    plt.legend()

    plt.savefig("outputs/loss_kl.png")
    plt.close()


def main(): 
    #Hyperparameters
    n_epochs = 30
    batch_size = 30
    latent_dim = 256
    learning_rate = 0.00022


    #(reusing some code from assignment 1)
    set_seed(2026)
    set_plot_settings()
    set_gpu()
    cache = True

    class_map = {
        "Happy": [1,0,0,0],
        "Sad": [0,1,0,0],
        "Surprised": [0,0,1,0],
        "Mad": [0,0,0,1]
    }
    num_classes = len(class_map)


    # Directory where the data are stored
    data_dir = "/import/course/5dv236/vt26/AffectNet/"
    # Create dataloaders
    train_ds, val_ds, test_ds = create_dataloaders(data_dir, class_map, batch_size, cache)

    X, y = train_ds[0] #get first batch of training data in X and y

    #some visualizations
    print(X.shape)
    print(y.shape)
    print(X.dtype)
    print(y.dtype)
    print(X.min(), X.max())
    print(y[:5])

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)


    #build encoder and decoder
    encoder = build_encoder(num_classes, latent_dim)
    decoder = build_decoder(num_classes, latent_dim)
    #build and compile model
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    #train  and save model
    history = vae.fit(train_ds, validation_data=val_ds, epochs=n_epochs)   

    #encoder.save("encoder_a3.keras")
    #decoder.save("decoder_a3.keras")

    #plot losses and save
    plot_all_loss(history)

    
    #generate n reconstructions and save
    n=4
    for emotion_name in class_map.keys():
        X_selected = []
        y_selected = []

        emotion_onehot = class_map[emotion_name]

        for i in range(len(X)):
            if np.array_equal(y[i], emotion_onehot):
                X_selected.append(X[i])
                y_selected.append(y[i])
            if len(X_selected) >= n:
                break

        X_selected = tf.convert_to_tensor(X_selected, dtype=tf.float32)
        y_selected = tf.convert_to_tensor(y_selected, dtype=tf.float32)

        xp = vae([X_selected, y_selected], training=False)

        plt.figure(figsize=(2*n+1, 4))
        plt.title(f"Reconstructed {emotion_name} images")
        plt.axis("off")

        for i in range(n):
            # original
            plt.subplot(2, n, i+1)
            plt.imshow(X_selected[i])
            plt.axis("off")

            # reconstructed
            plt.subplot(2, n, i+n+1)
            plt.imshow(xp[i])
            plt.axis("off")

        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/reconstructions_{emotion_name}.png")
        plt.close()


    #TASK 2A
    #find first happy
    #as the first and second happy image is dark, i will select the third happy image
    cont=0
    for i in range(len(y)):
        if np.array_equal(y[i], class_map["Happy"]):
            happy_index = i
            cont+=1 
            if cont>=3:
                break

    X = X[happy_index:happy_index+1]
    y = y[happy_index:happy_index+1]

    mean, logvar, z =encoder([X, y], training=False)

    #create a Sad condition
    y_sad = tf.constant([[0, 1, 0, 0]], dtype=tf.float32)

    #sample 5 latent vectors
    epsilon = tf.random.normal(shape=(5, latent_dim))
    sigma = tf.exp(0.5*logvar)

    z_samples = mean + sigma*epsilon

    #5 sad labels
    y_sad_samples = tf.repeat(y_sad, repeats=5, axis=0)

    generated = decoder([z_samples, y_sad_samples], training=False)

    plt.figure(figsize=(20, 3))
    plt.title("Happy latent vector and Sad label")

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(generated[i])
        plt.axis("off")

    plt.savefig("outputs/task2a.png")
    plt.close()

    #TASK 2B
    happy_means = []
    sad_means = []

    #get the mean latent vector for happy and sad
    for X_batch, y_batch in train_ds:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        mean, logvar, z = encoder([X_batch, y_batch], training=False)

        for i in range(len(y_batch)):
            if np.array_equal(y_batch[i], class_map["Happy"]):
                happy_means.append(mean[i])

            if np.array_equal(y_batch[i], class_map["Sad"]):
                sad_means.append(mean[i])

    happy_means = tf.stack(happy_means)
    sad_means = tf.stack(sad_means)
    happy_mean = tf.reduce_mean(happy_means, axis=0)
    sad_mean = tf.reduce_mean(sad_means, axis=0)

    #find the direction from happy to sad
    direction = sad_mean -happy_mean
    step = direction/9

    z_points = []

    for i in range(10):
        z_point = happy_mean + i*step
        z_points.append(z_point)

    z_points = tf.stack(z_points)
    y_sad = tf.constant([class_map["Sad"]]*10, dtype=tf.float32)

    #decode the 10 points along the line
    generated = decoder([z_points, y_sad], training=False)

    plt.figure(figsize=(20, 3))
    plt.title("From Happy to Sad with Sad label")

    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(generated[i])
        plt.axis("off")

    plt.savefig("outputs/task2b.png")
    plt.close()


    vae.save("mymodel_a3.keras")

    

if __name__ == "__main__":
    main()