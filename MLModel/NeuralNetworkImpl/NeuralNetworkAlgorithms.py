#Testing is pending
import numpy as np 
from Activation import ActivationFunctions
import random
from CostFunction import CostFunction

def backpropagation(X , Y, weights, biases, n_layers):
    gradient_w = [np.zeros(w.shape) for w in weights]
    gradient_b = [np.zeros(b.shape) for b in biases]

    activation = np.asarray(X)
    activation = activation.reshape(activation.shape[0],1)
    activations = [activation]
    zs = []
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = ActivationFunctions.sigmoid(z)
        activations.append(activation)

    delta = activations[-1] - Y



    gradient_b[-1] = delta
    gradient_w[-1] = np.dot(delta , activations[-2].transpose())

    for l in range(2, n_layers):
        z = zs[-l]
        sd = ActivationFunctions.sigmoidDerivative(z)
        delta = np.multiply(np.dot(weights[-l+1].transpose(), delta) , sd)
        gradient_b[-l] = delta
        gradient_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (gradient_b, gradient_w)

def updateMiniBatch(mini_batch, learning_rate, weights, biases, n_layers):
    gradient_b = [np.zeros(b.shape) for b in biases]
    gradient_w = [np.zeros(w.shape) for w in weights]

    for x,y in mini_batch:
        delta_b, delta_w = backpropagation(x, y, weights,biases,n_layers)
        gradient_b = [gb+db for gb, db in zip(gradient_b, delta_b)]
        gradient_w = [gw+dw for gw, dw in zip(gradient_w, delta_w)]
    weights = [w-(learning_rate/len(mini_batch))*gw
                    for w, gw in zip(weights, gradient_w)]
    biases = [b-(learning_rate/len(mini_batch))*gb
                   for b, gb in zip(biases, gradient_b)]
    return weights,biases

# training_data is a list of tuples - [(x,y),(x2,y2),.....]
def stochasticGradientDescent(training_data, epochs, mini_batch_size, learning_rate, weights, biases, n_layers):
    m = len(training_data)
    for i in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, m, mini_batch_size)]
        for mini_batch in mini_batches:
            weights,biases = updateMiniBatch(mini_batch, learning_rate, weights, biases, n_layers)
        print("Epoch " +  str(i) + " complete")
        
    return weights,biases

    





		