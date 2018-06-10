
import gzip
from mnist import MNIST
import numpy as np

def load_data():
    mndata = MNIST('../../Data')
    mndata.gz = True
    training_images , training_label = mndata.load_training()
    testing_images , testing_label = mndata.load_testing()
    training_label = [vectorized_result(j) for j in training_label]
    testing_label = [vectorized_result(j) for j in testing_label]
    training_data = [x for x in zip(training_images,training_label)]
    testing_data = [x for x in zip(testing_images , testing_label)]
    return training_data, testing_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
