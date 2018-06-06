#main implementation of neural network

import numpy as np
import CostFunction
import Activation

class NeuralNetwork:
	def __init__(self , n_input, n_output, n_layers, n_neurons):
		self.n_input = n_input
		self.n_output = n_output
		self.n_layers = n_layers
		self.n_neurons = n_neurons
		self.weights = [] # weights contains biases from each layer as well
		self.initializeWeights()

	def initializeWeights(self):
		# He-et-al Initialization
		weight = np.random.randn(self.n_neurons , self.n_input + 1) * np.sqrt(2 / self.n_neurons)  
		self.weights.append(weight)


		for i in range(self.n_layers - 1):
			weight = np.random.randn(self.n_neurons , self.n_neurons + 1) * np.sqrt(2 / self.n_neurons)
			self.weights.append(weight)

		weight = np.random.randn(self.n_output , self.n_neurons + 1) * np.sqrt(2 / self.n_output)
		self.weights.append(weight) 

