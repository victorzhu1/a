#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

#load the dataset with the images of fashion items
(x_train, _), (x_test, _) = fashion_mnist.load_data()

#train the dataset with these images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#add random noise to these images
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

# the "clip by value" function "clips" the values in x_train_noisy to between 0 and 1, so values 
# below 0 are set by default to 0, and values above 1 set by default to 1
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# autoencoder class, encoder uses Conv2D to downscale the 28x28 original image to a 7x7 one.
# then the decoder upscales it to a new reconstructed image
#not sure what the activation, padding, and strides are
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

# from what i read "adam" is a gradient descent method (i rememebr this from the ap research project)
# also this uses mean squared error, i think that is what it uses to calculate loss
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# runs the autoencoder with 10 epochs (from what i can tell an epoch is one training iteration)
# the shuffle = true means that it will shuffle the input of data for that specific epoch which
# helps with accuracy and also to prevent overfitting.
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# all of this below plots the original noisy images and then the reconstructed images.
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()