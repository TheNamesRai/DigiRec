#main implementation of neural network

import numpy as np
import CostFunction
from Activation import ActivationFunctions
import NeuralNetworkAlgorithms
import random

class NeuralNetwork:
	def __init__(self , n_input, n_output, n_neurons):
		self.n_input = n_input
		self.n_output = n_output
		self.n_layers = len(n_neurons)
		self.n_neurons = n_neurons

		self.sizes = [n_input]
		self.sizes.extend(n_neurons)
		self.sizes.append(n_output)

		# He-et-al Initialization
		#self.weights = [np.random.randn(y , x) * np.sqrt(2 / y) for x,y in zip(self.sizes[:-1] , self.sizes[1:])] 
		self.weights = [np.random.randn(y , x) for x,y in zip(self.sizes[:-1] , self.sizes[1:])]
		self.biases = [np.random.randn(y , 1) for y in self.sizes[1:]]


	def feedForward(self , x):
		x = np.asarray(x)
		x = x.reshape(x.shape[0],1)
		for b,w in zip(self.biases, self.weights):
			x = ActivationFunctions.sigmoid(np.dot(w,x) + b)
		return x

	def train(self, training_data , epochs, mini_batch_size, learning_rate):
		self.stochasticGradientDescent(training_data,
									epochs, mini_batch_size, learning_rate)

	def load_file(self):
		self.weights = np.load('weights.npy')
		self.biases = np.load('biases.npy')


	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for (x, y) in test_data]
		print(test_results[0:10])
		acc = (sum(int(x == y) for (x, y) in test_results) / len(test_results))
		print("Accuracy : " + str(acc) )




	def backpropagation(self,X , Y):
	    gradient_w = [np.zeros(w.shape) for w in self.weights]
	    gradient_b = [np.zeros(b.shape) for b in self.biases]

	    activation = np.asarray(X)
	    activation = activation.reshape(activation.shape[0],1)
	    activations = [activation]
	    zs = []
	    for b, w in zip(self.biases, self.weights):
	        z = np.dot(w, activation)+b
	        zs.append(z)
	        activation = ActivationFunctions.sigmoid(z)
	        activations.append(activation)

	    delta = (activations[-1] - Y) * ActivationFunctions.sigmoidDerivative(zs[-1])



	    gradient_b[-1] = delta
	    gradient_w[-1] = np.dot(delta , activations[-2].transpose())

	    for l in range(2, self.n_layers + 2):
	        z = zs[-l]
	        sd = ActivationFunctions.sigmoidDerivative(z)
	        delta = np.dot(self.weights[-l+1].transpose(), delta) * sd
	        gradient_b[-l] = delta
	        gradient_w[-l] = np.dot(delta, activations[-l-1].transpose())
	    return (gradient_b, gradient_w)

	def updateMiniBatch(self, mini_batch, learning_rate):
	    gradient_b = [np.zeros(b.shape) for b in self.biases]
	    gradient_w = [np.zeros(w.shape) for w in self.weights]

	    for x,y in mini_batch:
	        delta_b, delta_w = self.backpropagation(x, y)
	        gradient_b = [gb+db for gb, db in zip(gradient_b, delta_b)]
	        gradient_w = [gw+dw for gw, dw in zip(gradient_w, delta_w)]
	    self.weights = [w-(learning_rate/len(mini_batch))*gw
	                    for w, gw in zip(self.weights, gradient_w)]
	    self.biases = [b-(learning_rate/len(mini_batch))*gb
	                   for b, gb in zip(self.biases, gradient_b)]


	# training_data is a list of tuples - [(x,y),(x2,y2),.....]
	def stochasticGradientDescent(self,training_data, epochs, mini_batch_size, learning_rate):
	    m = len(training_data)
	    for i in range(epochs):
	        random.shuffle(training_data)
	        mini_batches = [
	            training_data[k:k+mini_batch_size]
	            for k in range(0, m, mini_batch_size)]
	        for mini_batch in mini_batches:
	            self.updateMiniBatch(mini_batch, learning_rate)
	        print("Epoch " +  str(i) + " complete")
	        self.evaluate(training_data)



		