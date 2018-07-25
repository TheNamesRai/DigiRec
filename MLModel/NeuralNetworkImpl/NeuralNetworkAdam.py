#Implementation is in inprogress

import numpy as np
import CostFunction
from Activation import ActivationFunctions
import NeuralNetworkAlgorithms
import random

class NeuralNetworkAdam:
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

	def train(self, training_data , epochs, mini_batch_size, learning_rate, gamma):
		self.sgdMomentum(training_data,
									epochs, learning_rate, gamma)

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

	    #delta = (activations[-1] - Y) * ActivationFunctions.sigmoidDerivative(zs[-1])
	    delta = (activations[-1] - Y)



	    gradient_b[-1] = delta
	    gradient_w[-1] = np.dot(delta , activations[-2].transpose())

	    for l in range(2, self.n_layers + 2):
	        z = zs[-l]
	        sd = ActivationFunctions.sigmoidDerivative(z)
	        delta = np.dot(self.weights[-l+1].transpose(), delta) * sd
	        gradient_b[-l] = delta
	        gradient_w[-l] = np.dot(delta, activations[-l-1].transpose())
	    return (gradient_b, gradient_w)


	#-----------------------------------------------------------------------------------------



	def update(self, training_data, learning_rate,M_w, M_b, R_w, R_b,t, gamma = 0.9, gamma2 = 0.999):
		gradient_b = [np.zeros(b.shape) for b in self.biases]
		gradient_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in training_data:
			delta_b, delta_w = self.backpropagation(x,y)
			gradient_b = [gb+db for gb, db in zip(gradient_b, delta_b)]
			gradient_w = [gw+dw for gw, dw in zip(gradient_w, delta_w)]
		M_w = [(m_w * gamma) + (1.0 - gamma)*gw for gw,m_w in zip(gradient_w,M_w)]
		M_b = [(m_b * gamma) + (1.0 - gamma)*gb for gb,m_b in zip(gradient_b,M_b)]
		R_w = [(r_w * gamma2) + (1.0 - gamma2)*(gw**2) for gw,r_w in zip(gradient_w,R_w)]
		R_b = [(r_b * gamma2) + (1.0 - gamma2)*(gb**2) for gb,r_b in zip(gradient_b,R_b)]
		m_w_hat = [m_w / (1.0 - gamma**t) for m_w in M_w]
		m_b_hat = [m_b / (1.0 - gamma**t) for m_b in M_b]
		r_w_hat = [r_w / (1.0 - gamma2**t) for r_w in R_w]
		r_b_hat = [r_b / (1.0 - gamma2**t) for r_b in R_b]
		self.weights = [w-(learning_rate*mw)/(len(training_data) * (np.sqrt(rw) + np.exp(-8)))
		                for w, mw,rw in zip(self.weights, m_w_hat,r_w_hat)]
		self.biases = [b-(learning_rate*mb)/(len(training_data) * (np.sqrt(rb) + np.exp(-8)))
		               for b, mb,rb in zip(self.biases, m_b_hat, r_b_hat)]
		return M_w, M_b, R_w, R_b

	# training_data is a list of tuples - [(x,y),(x2,y2),.....]
	def sgdMomentum(self, training_data, epochs, learning_rate, gamma):
		M_b = [np.zeros(b.shape) for b in self.biases]
		M_w = [np.zeros(w.shape) for w in self.weights]
		R_b = [np.zeros(b.shape) for b in self.biases]
		R_w = [np.zeros(w.shape) for w in self.weights]
		m = len(training_data)
		for i in range(epochs):
			random.shuffle(training_data)
			M_w, M_b, R_w, R_b = self.update(training_data, learning_rate, M_w, M_b, R_w, R_b,i+1)
			print("Epoch " + str(i) + " complete")
			self.evaluate(training_data)





		