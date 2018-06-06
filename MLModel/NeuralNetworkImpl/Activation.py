import numpy as np


class ActivationFunctions:
	def sigmoid(x):
		return 1.0 / (1.0 + np.exp(-x))

	def sigmoidDerivative(x):
		return np.multiply(sigmoid(x) , (1 - sigmoid(x)))

	def tanh_prime(x):
		return 1 - np.tanh(x)**2



