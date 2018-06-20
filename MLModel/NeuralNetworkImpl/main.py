import numpy as np 
import time
from Activation import ActivationFunctions
from NeuralNetwork import NeuralNetwork
from NeuralNetworkAdam import NeuralNetworkAdam
from CostFunction import CostFunction
import Data
import time

start = time.time()

#list of number of neurons in hidden layers

n_neurons = [100]
n_input = 784

# output layer
n_output = 10

#training examples
n_sample = 60000

#hyper parameters
learning_rate = 4
momentum = 0.9

epochs = 150
mini_batch_size = 10

training_data, testing_data = Data.load_data()



nn = NeuralNetwork(n_input, n_output,  n_neurons)



#print(nn.feedForward(training_data[0][0]))

nn.train(training_data, epochs, 60000, learning_rate , momentum)
np.save('weights.npy' , nn.weights)
np.save('biases.npy' , nn.biases)
#nn.load_file()

nn.evaluate(testing_data)


end = time.time()
print(end -start)