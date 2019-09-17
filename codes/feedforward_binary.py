# -*- coding: utf-8 -*-
"""deepLearning_feedforward.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/hfarruda/deeplearningtutorial/blob/master/deepLearning_feedforward.ipynb

#Feedforward networks

This example is part of the [*Deep Learning Tutorial*](https://github.com/hfarruda/deeplearningtutorial), authored by Henrique F. de Arruda, Alexandre Benatti, César Comin, and Luciano da Fontoura Costa.  This code is not suitable for other data and/or applications, which will require modifications in the structure and parameters. These codes have absolutely no warranty. 

If you publish a paper related on this material, please cite:

H. F. de Arruda, A. Benatti, C. H. Comin, L.  da  F.  Costa,  "Learning Deep Learning (CDT-15)," 2019.

## Binary Classification
This is the first example of deep learning implementation, in which we address binary classification of wine data. In this example, we consider one feedforward network with 5 hidden layers and with 30 neurons in each layer.  The provided networks were built only for didactic purposes and are not appropriate for real applications.

First of all, we import the necessary libraries. Here we opt for using Keras (using TensorFlow backend).
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""If you have a GPU, you can use the following code to allocate processing into it.  Otherwise, proceed to (*)."""

import tensorflow as tf 
from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())

number_of_cpu_cores = 8
config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': number_of_cpu_cores}) 
session = tf.Session(config=config) 
keras.backend.set_session(session)

"""Here, we use the Wine dataset. It is available at Sklearn library on [sklearn-datasets-wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). For more information [wine-UCI](https://archive.ics.uci.edu/ml/datasets/Wine).
Because this dataset comprises three classes and here we exemplify a binary classification, we considered only the two first classes.
"""

wine = load_wine()
data = wine['data']
target = wine['target']
target_names = wine['target_names'] 

#The selected items are stored in the variable called "hold".
hold = np.argwhere(target!=2).T[0]
data = data[hold]
target = target[hold]
target_names = target_names[0:1]

#Here, we divide our dataset into training and test sets.
test_size = 0.25 #fraction 
training_data,test_data,training_target,test_target = train_test_split(data, 
                                                    target, test_size=test_size)

"""In the following, we configure the neuronal network. It is not necessary to include bias because this parameter is set as true by default."""

#Set of parameters
input_dim = data.shape[1]
kernel_initializer = 'random_uniform'
bias_initializer='zeros'
activation_function_hidden = 'relu'
activation_function_output = 'sigmoid'
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']
number_of_layers = 5
number_of_units_hidden = 30
number_of_units_output = 1
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

"""In order to check the network topology, you can use the subsequent command."""

ff_model.summary()

"""Another option is to visualize the topology as a figure."""

#Saving the resultant figure as 'ff_model.png'.
plot_model(ff_model, to_file='ff_model.png', show_shapes=True, 
           show_layer_names=True)

"""Next, we train the network"""

batch_size = 10
epochs = 200
ff_model.fit(training_data,training_target, batch_size = batch_size, 
             epochs = epochs)

"""In order to create an application, it is possible to save the network and the respective trained weights as follows."""

#Saving the network model
ff_model_json = ff_model.to_json()
with open('ff_model.json', 'w') as file:
    file.write(ff_model_json)

#Saving weights
ff_model.save_weights('ff_model.h5')

"""The following code can be employed to open a pre-trained model."""

with open('ff_model.json', 'r') as file:
    ff_model_json = file.read()

ff_model = model_from_json(ff_model_json)
ff_model.load_weights('ff_model.h5')

"""There are different analysis that can account for the quality of the results. Here, we consider only the measurement of accuracy."""

predictions = ff_model.predict(test_data)
#Because it is a binary classification, we consider the values higher than 0.5 
#as being part of class 1 otherwise 0.
predictions = (predictions > 0.5)
accuracy = accuracy_score(test_target, predictions)
print("Accuracy =", accuracy)


"""## License

This Deep Learning Tutorial is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0) International License.
"""