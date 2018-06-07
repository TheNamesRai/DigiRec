#Testing is pending
import numpy as np 
from Activation import ActivationFunctions

class NeuralNetworkGradient:
	def backpropagation(n_samples , X , n_input, Y, n_output, weights , n_layers, n_neurons):
		activations = []
		