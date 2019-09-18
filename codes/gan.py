# -*- coding: utf-8 -*-
"""deepLearning_GAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CbRNDN25uaN2WCeyklwPkbZZkXM_IDge

#  Generative Adversarial Networks
This example is part of the [*Deep Learning Tutorial*](https://github.com/hfarruda/deeplearningtutorial), authored by Henrique F. de Arruda, Alexandre Benatti, César Comin, and Luciano da Fontoura Costa. This code is not suitable for other data and/or applications, which will require modifications in the structure and parameters. This code has absolutely no warranty.

If you publish a paper related on this material, please cite:

H. F. de Arruda, A. Benatti, C. H. Comin, L. da F. Costa, "Learning Deep Learning (CDT-15)," 2019.

It was elaborated to create a network that can generate handwritten characters automatically.


First of all, we import the necessary libraries. Here we opt for using Keras (using TensorFlow backend).
"""

import numpy as np
import pandas as pd 
import keras
from keras.models import Sequential, model_from_json
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.layers import InputLayer, Dense, Flatten, Reshape, Input, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model,Sequential
from keras.regularizers import L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2

"""If you have a GPU, you can use the following code to allocate processing into it.  Otherwise, proceed to (*)."""

import tensorflow as tf 
from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())

number_of_cpu_cores = 8
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': number_of_cpu_cores}) 
session = tf.Session(config=config) 
keras.backend.set_session(session)

"""(*) In this example we used the MNIST database in which it is composed by grayscale images of the 10 handwritten digits. It is available at Keras library on [keras-datasets](https://keras.io/datasets/).

The following command is used to load the data set.
"""

(train_data_raw, train_target_raw), (_, _) = mnist.load_data()

"""Because this code consumes too much of processing time, here we considered only the zeros and ones."""

train_data = [img for i, img in enumerate(train_data_raw) 
              if train_target_raw[i] == 0 or train_target_raw[i] == 1]
train_data = np.array(train_data)

"""In order to visualize a given figure, the following code can be executed."""

image_id = 1000
plt.figure(figsize = (1,1))
plt.imshow(train_data[image_id], cmap='gray')
plt.title("Test image: " + str(image_id))
#plt.axis('off')
plt.show()

"""Definition of the used variables."""

input_shape = train_data.shape[1::]
activation_output_generator = 'sigmoid'
activation_output_discrimninator = 'sigmoid'
input_dim = 50
number_of_epochs = 1000
batch_size = 100
train_data = train_data.astype('float32') / 255

"""In the following, we present the generator model."""

generator_model = Sequential()

generator_model.add(Dense(units=64,input_dim = input_dim, 
                          kernel_regularizer = L1L2(1e-5, 1e-5)))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.3))

generator_model.add(Dense(units=128, kernel_regularizer = L1L2(1e-5, 1e-5)))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.3))

generator_model.add(Dense(units=256, kernel_regularizer = L1L2(1e-5, 1e-5)))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.3))

generator_model.add(Dense(units = input_shape[0] * input_shape[1], 
                          activation = activation_output_generator))

generator_model.add(Reshape(input_shape))

generator_model.compile(loss='binary_crossentropy', optimizer="adam")

"""The summary of the generator model is shown by employing the following code."""

generator_model.summary()

"""The following code represents the discriminator model."""

discriminator_model = Sequential()
discriminator_model.add(InputLayer(input_shape = input_shape))
discriminator_model.add(Flatten())

discriminator_model.add(Dense(units=256,kernel_regularizer = L1L2(1e-5, 1e-5)))
discriminator_model.add(LeakyReLU(alpha=0.3))
discriminator_model.add(Dropout(0.2))


discriminator_model.add(Dense(units=128,kernel_regularizer = L1L2(1e-5, 1e-5)))
discriminator_model.add(LeakyReLU(alpha=0.3))
discriminator_model.add(Dropout(0.2))

discriminator_model.add(Dense(units=64,kernel_regularizer = L1L2(1e-5, 1e-5)))
discriminator_model.add(LeakyReLU(alpha=0.3))

discriminator_model.add(Dense(units=1, 
                              activation = activation_output_discrimninator))

discriminator_model.compile(loss='binary_crossentropy', 
                            optimizer = "adam")

"""The summary of the discriminator model is shown by using the following code."""

discriminator_model.summary()

"""The following code incorporates the complete gan model."""

gan_input = Input(shape = (input_dim,))
gan_output= discriminator_model(generator_model(gan_input))
gan = Model(inputs = gan_input, outputs = gan_output)
gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')

"""The summary of the gan model is shown by using the following code."""

gan.summary()

"""Next, we train the GAN."""

y = np.ones(batch_size)

#Parameters of the noise distribution
mu = 0
sigma = 1

#We created this array to avoid number repetitions
train_indices = np.arange(train_data.shape[0])
np.random.shuffle(train_indices)

#Here we define the labels used to train the gan
train_labels = np.zeros(2*batch_size,dtype = int)
train_labels[0:batch_size] = 1#generated images

for epoch in range(number_of_epochs):
    print("\rEpoch:", epoch + 1, "of", number_of_epochs, end = '')
    for _ in range(batch_size):
        input_noise = np.random.normal(loc = mu, scale = sigma, 
                                       size = [batch_size, input_dim])
        generated_images = generator_model.predict(input_noise)
        np.random.shuffle(train_indices)
        image_batch = train_data[train_indices[0:batch_size]]
        train_images = np.concatenate((image_batch, generated_images))
        #Training the discriminator
        discriminator_model.trainable = True
        discriminator_model.train_on_batch(train_images, train_labels)
        #Training the gan
        discriminator_model.trainable = False
        train_noise = np.random.normal(loc = mu, scale = sigma, 
                                       size =  [batch_size, input_dim])
        gan.train_on_batch(train_noise, y)

  #In order to visualize the training progress, we employ the following code.
    if epoch % 100 == 0:
        n_examples = 10
        scale_image = 1 * n_examples
        noise= np.random.normal(loc = mu, scale = sigma, 
                                size = (n_examples, input_dim))
        generated_images = generator_model.predict(noise)
        n_pixels = generated_images.shape[1]
        n_pixels_col = np.int(np.sqrt(n_pixels))
        fig, axes = plt.subplots(1,n_examples, 
                                 figsize = (scale_image, 
                                            scale_image * n_examples))
    for i in range(generated_images.shape[0]):
        axes[i].imshow(generated_images[i], cmap = "gray")
        axes[i].axis('off')
    plt.show()
print("")

"""In order to generate the figures the following code can be employed."""

n_examples = 5
scale_image = 5
noise= np.random.normal(loc = mu, scale = sigma, size = (n_examples, input_dim))
generated_images = generator_model.predict(noise)

fig, axes = plt.subplots(1,n_examples,
                         figsize = (scale_image, scale_image * n_examples))
for i in range(generated_images.shape[0]):
    axes[i].imshow(generated_images[i], cmap = "gray")
    axes[i].axis('off')

plt.show()

"""## License

This Deep Learning Tutorial is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0) International License.
"""