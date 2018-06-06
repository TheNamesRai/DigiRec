# return cost as well as gradient
# backprop

# H : our prediction : n_samples by n_output matix
# Y : Actual Value : n_sample by n_output matrix
def cost(n_samples, H, Y, n_output, regularization = False, weights = None, reg_rate = None):
	total_cost = 0

	#cross entropy loss function
	total_cost = (np.multiply(-Y , np.log(H)) - np.multiply((1-Y) , np.log(1-H))) / n_samples

	if(regularization == True and weights != None and reg_rate != None):
		pass


	return total_cost