import numpy as np 
import time
import Activation
from NeuralNetwork import NeuralNetwork

#number of neurons in hidden layers
n_hidden = 10
n_in = 10

# output layer
n_output = 10

#training examples
n_sample = 300

#hyper parameters
learning_rate = 0.01
momentum = 0.9


m1 = np.array([[2,4] , [3,4]])
m2 = np.array([[1,2],[2,3]])

print(np.matmul(m1,m2))

print(np.transpose(m1))


d = NeuralNetwork(3,4)
