import numpy as np 
import time
from Activation import ActivationFunctions
from NeuralNetwork import NeuralNetwork
from CostFunction import CostFunction

#number of neurons in hidden layers
n_neurons = 3
n_input = 4

# output layer
n_output = 2

#training examples
n_sample = 300

#hyper parameters
learning_rate = 0.01
momentum = 0.9

#number of hidden layers
n_layers = 1 # n_layers >= 1


m1 = np.array([[2,5 , 6] , [3,6 , 4]])
m2 = np.array([[1,4],[7,8]])

print(np.square(m1));



nn = NeuralNetwork(n_input, n_output, n_layers, n_neurons)

nn.printWeights()