# -*- coding: utf-8 -*-
"""deepLearning_feedforward.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/hfarruda/deeplearningtutorial/blob/master/deepLearning_feedforward.ipynb

#Feedforward networks

This example is part of the [*Deep Learning Tutorial*](https://github.com/hfarruda/deeplearningtutorial), authored by Henrique F. de Arruda, Alexandre Benatti, César Comin, and Luciano da Fontoura Costa.  This code is not suitable for other data and/or applications, which will require modifications in the structure and parameters. These codes have absolutely no warranty. 

If you publish a paper related on this material, please cite:

H. F. de Arruda, A. Benatti, C. H. Comin, L.  da  F.  Costa,  "Learning Deep Learning (CDT-15)," 2019.

##Multiclass Classification
In this example, we illustrate a multiclass classification through a wine dataset, in which there are three classes, which were defined according to their regions. We employed the same dataset presented above, but here we considered the three classes. To do so, we use the *softmax* activation function.

First of all, we import the necessary libraries. Here we opt for using Keras (using TensorFlow backend).
"""

import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

"""If you have a GPU, you can use the following code to allocate processing into it.  Otherwise, proceed to (*)."""

import tensorflow as tf 
from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())

number_of_cpu_cores = 8
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': number_of_cpu_cores}) 
session = tf.Session(config=config) 
keras.backend.set_session(session)

"""(*) In this example the dataset used is Wine. It is available at Sklearn library on [sklearn-datasets-wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). For more information [wine-UCI](https://archive.ics.uci.edu/ml/datasets/Wine).

These data show the results of a chemical analysis of wines grown in Italy, derived from three different cultivars in the same region, and can be loaded as follows.
"""

wine = load_wine()
data = wine['data']
target = wine['target']
target_names = wine['target_names'] 

label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)
target_one_hot_encoding = np_utils.to_categorical(target)

#Here, we divide our dataset into training and test sets.
test_size = 0.25 #fraction 
training_data,test_data,training_target,test_target = train_test_split(data, 
                                  target_one_hot_encoding, test_size=test_size)

"""In the following, we configure the neuronal network. It is not necessary to include bias because this parameter is set as true by default."""

#Set of parameters
input_dim = data.shape[1]
kernel_initializer = 'random_uniform'
bias_initializer='zeros'
activation_function_hidden = 'relu'
activation_function_output = 'softmax'
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
number_of_layers = 5
number_of_units_hidden = 30
number_of_units_output = len(set(target_names))
dropout_percentage = 0.25


#Creating model
ff_model = Sequential()
ff_model.add(Dense(units = number_of_units_hidden, 
                   activation = activation_function_hidden, 
                   kernel_initializer = kernel_initializer, 
                   input_dim = input_dim))

for i in range(number_of_layers-1):
  #Inserting a dense hidden layer
  ff_model.add(Dense(units = number_of_units_hidden, 
                     activation = activation_function_hidden, 
                     kernel_initializer = kernel_initializer, 
                     input_dim = number_of_units_hidden))
  #Inserting dropout
  ff_model.add(Dropout(dropout_percentage))

ff_model.add(Dense(units = number_of_units_output, 
                   activation = activation_function_output))
ff_model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
ff_model.summary()

"""The training step is executed as follows."""

batch_size = 10
epochs = 250
ff_model.fit(training_data,training_target, batch_size = batch_size,
             epochs = epochs)

"""Because there are three classes, we show the classification results through a confusion matrix."""

predictions = ff_model.predict(test_data)

found_target = predictions.argmax(axis=1)
categorical_test_target = test_target.argmax(axis=1)

accuracy = accuracy_score(categorical_test_target, found_target)
print("Accuracy =", accuracy)

print("Confusion matrix:")
matrix = confusion_matrix(found_target,categorical_test_target)
print(matrix)

"""## License

This Deep Learning Tutorial is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0) International License.
"""