"""
This script includes the basic structure of a genetic algorithm structure,
which is going to be applied to a feedforward neural network and CasPer techniques.

Specifically, the genetic algorithm is used for optimisation in deciding
hyper-parameters and feature selection.
"""

import numpy as np
import matplotlib.pyplot as plt

class GA_model():
    def __init__(self, DNA_size, pop_size, cross_rate, mutation_rate, n_generations, train_test):
        # acquire GA settings
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.train_test = train_test
        self.nn = True if DNA_size==44 else False    # whether it's NN or CasPer

    def train(self, plot=False):
        # initialise population DNA
        pop = np.random.randint(0, 2, (self.pop_size, self.DNA_size))

        # start training
        all_highest_correctness = []
        for t in range(self.n_generations):
            print("-------- generation {} --------".format(str(t+1)))
            # convert binary DNA to hyper parameters settings
            hyper_pop = self.translateDNA(pop)
            correctness = self.get_fitness(hyper_pop)
            print("Highest correctness: ", max(correctness))
            print("Corresponding hyper parameters settings: ", self.translateDNA(np.expand_dims(pop[np.argmax(correctness),:],axis=0)))
            print("---------------------------------------\n")
            all_highest_correctness.append(max(correctness))

        if plot:
            line = plt.plot(range(1,self.n_generations+1), all_highest_correctness)
            plt.setp(line, color='g', linewidth=2.0)
            plt.xlabel('number of generation')
            plt.ylabel('best test accuracy rate among population (%)')
            plt.title('best DNA performance over time')
            plt.show()

    # define fitness function, here basically is the average accuracy rate in test data set
    def get_fitness(self, hyper_pop):
        if self.nn:
            # extract hyper parameters
            features_pop = hyper_pop[0]
            hidden_size_pop = hyper_pop[1]
            num_epochs_pop = hyper_pop[2]
            learning_rate_pop = hyper_pop[3]
            # store all highest accuracy rate
            correctness = []
            for p in range(self.pop_size):
                hyper = [features_pop[p], hidden_size_pop[p], num_epochs_pop[p], learning_rate_pop[p]]
                print("{}/{}".format(p+1, self.pop_size))
                print(hyper)
                highest_test_correctness, train_correctness, test_loss, train_loss = self.train_test(hyper)
                correctness.append(highest_test_correctness)
        else:
            # extract hyper parematers
            features_pop = hyper_pop[0]
            num_neurons = hyper_pop[1]
            p_value = hyper_pop[2]
            # store all highest accuracy rate
            correctness = []
            for p in range(self.pop_size):
                hyper = [features_pop[p], num_neurons[p], p_value[p]]
                print("{}/{}".format(p + 1, self.n_generations))
                print(hyper)
                highest_test_correctness, train_correctness, test_loss, train_loss = self.train_test(hyper)
                correctness.append(highest_test_correctness)
        print(correctness)
        return correctness

    # convert binary DNA to decimals, and transform into meaningful hyper parameters accordingly
    def translateDNA(self, pop):
        # extract feature selection
        hyper = [pop[:, :20]]

        # extract other hyper parameters
        if self.nn:     # if it's a feedforward neural network
            hidden_size = pop[:, 20:26,]
            num_epochs = pop[:, 26:34]
            learning_rate = pop[:, 34:44]
            hidden_size = binary_to_decimal(gray_to_binary(hidden_size)) + 1
            num_epochs = binary_to_decimal(gray_to_binary(num_epochs)) + 10
            learning_rate = (binary_to_decimal(gray_to_binary(learning_rate))+1)/10000
            hyper.extend([hidden_size, num_epochs, learning_rate])
        else:   # if it's CasPer
            num_neurons = pop[:, 20:24]
            p_value = pop[:, 24:27]
            num_neurons = binary_to_decimal(gray_to_binary(num_neurons)) + 1
            p_value = binary_to_decimal(gray_to_binary(p_value)) + 1
            hyper.extend([num_neurons, p_value])

        return hyper

    # define select function based on fitness value
    # population with higher fitness value ahs higher chance to be selected
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=fitness/fitness.sum())
        return pop[idx]

    # define gene crossover function
    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0,2,size=self.DNA_size).astype(np.bool)
            parent[cross_points] = pop[i, cross_points]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                child[point] = 1 if child[point] == 0 else 0
        return child

# convert binary to decimal, the input is a list of bits, output is a decimal number
def binary_to_decimal(binary):
    if isinstance(binary, list):
        binary = np.asarray(binary)
    return binary.dot(2 ** np.arange(binary.shape[1])[::-1])

# function to convert gray code list to binary list
def gray_to_binary(gray):
    binary = []
    for gray_i in gray:
        binary_i = [gray_i[0]]  # most significant bit of binary code is same as gray code
        for i in range(1, len(gray_i)):   # Compute remaining bits
            # if current bit is 0, append previous bit; otherwise append the invert
            binary_i.append(binary_i[i-1] if gray_i[i] == 0 else 0 if binary_i[i-1]==1 else 1)
        binary.append(binary_i)
    return binary