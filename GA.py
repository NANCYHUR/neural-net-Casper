"""
This script includes the basic structure of a genetic algorithm structure,
which is going to be applied to a feedforward neural network and CasPer techniques.

Specifically, the genetic algorithm is used for optimisation in deciding
hyper-parameters and feature selection.
"""

import numpy as np
import matplotlib.pyplot as plt

class GA_model():
    def __init__(self, DNA_size, pop_size, cross_rate, n_generations):
        # acquire GA settings
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.n_generations = n_generations

        # initialise population DNA
        pop = np.random.randint(0,2,(pop_size,DNA_size))

    def get_fitness(self, prediction):
        pass

    def translateDNA(self, pop):
        """
        for each DNA in pop,
        :param pop: a dataframe of size [pop_size, DNA_size]
        :return: translated feature selection (a 24-binary-element list), hidden layer size,
                    number of epochs and learning rate (in order)
        """
        hyper = [pop[:24]]

        return hyper


# function to convert gray code list to binary list
def graytoBinary(gray):
    binary = [gray[0]]  # most significant bit of binary code is same as gray code
    for i in range(1, len(gray)):   # Compute remaining bits
        # if current bit is 0, append previous bit; otherwise append the invert
        binary.append(binary[i-1] if gray[i] == 0 else 0 if binary[i-1]==1 else 1)
    return binary