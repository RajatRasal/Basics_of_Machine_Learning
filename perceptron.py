import numpy as np
from random import randint


class Perceptron:
    """ Simple Perceptron for classification """

    def __init__(self, rate=0.01 * randint(0, 100), iterations=50, random_gen_seed=1):
        """ Initialising hyperparameters of the the perceptron """
        # learning rate
        self.rate = rate
        # number of epochs
        self.iterations = iterations
        self.random_gen_seed = random_gen_seed

    def fit(self, x, y):
        """ Fits the training data
        x -> [[n_samples, n_features]] (training matrix)
        y -> [n_samples] (target vector).

        Fitting is the process of training the perceptron and then creating a
        model against which data can be classified.
        """
        self.errors = []

        # Since the size is an integer, a 1-D array filled with generated
        # values is returned by RandomState
        # Draws random samples from the Normal Distribution with st_dev of 0.01
        gen = np.random.RandomState(self.random_gen_seed)
        self.w = gen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(x, y):
                update = self.rate * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)

        return self

    def net_input(self, x):
        """ Calculate net input """
        # w[0] is the bias unit
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        """ Threshold function - returns the class label by calculating the
        net input """
        return np.where(self.net_input(x) >= 0, 1, -1)
