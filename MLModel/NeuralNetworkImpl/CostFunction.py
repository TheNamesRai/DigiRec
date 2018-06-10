# Testing is pending
import numpy as np
# H : our prediction : n_samples by n_output matix
# Y : Actual Value : n_samples by n_output matrix
# reg_rate : regularization rate
class CostFunction():

	def crossEntropy(n_samples, H, Y, n_output, regularization = False, weights = None, reg_rate = None):
		total_cost = 0

		#cross entropy loss function
		total_cost = (np.multiply(-Y , np.log(H)) - np.multiply((1-Y) , np.log(1-H))) / n_samples

		if(regularization == True and weights != None and reg_rate != None):
			reg = 0
			for i in range(len(weights)):
				reg += np.sum(np.square(weights[i]))
			reg = (reg * reg_rate) / (2 * n_samples)
			total_cost += reg


		return total_cost