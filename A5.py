# April 4, 2022
# Completing the tutorial found at https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import sample data set of clothing
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# labels for the images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# examine first image in training set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# data preprocessing
# scale pixel values to 0-1
train_images = train_images / 255.0 
test_images = test_images / 255.0

# make sure that things are working so far
# display first 25 training images with their labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1) # divide the output into smaller grids
    plt.xticks([])
    plt.yticks([])  # x and y locations
    plt.grid(False) # don't show grid lines
    plt.imshow(train_images[i], cmap=plt.cm.binary) # use binary colourmap
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# set up layers
model = tf.keras.Sequential([ # Sequential combines layers into a keras Model
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # turn 2d array of pixels into 1d array of 28 * 28 pixels
    tf.keras.layers.Dense(128, activation='relu'), # fully connected neural layer with 128 neurons
    tf.keras.layers.Dense(10)   # returns logits array of length 10
])