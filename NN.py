"""
This script builds a neural network for classifying forest supra-type based on
dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process, interpret_output
from GA import GA_model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

run_times = 20
plot_each_run = False

# GA settings
DNA_size = 48
pop_size = 20
cross_rate = 0.8
n_generations = 50

# fixed hyper parameters of NN
input_size = 20
output_size = 4
k_cross_validation = 5

# define loss function
loss_func = nn.MSELoss()


# get train data, used in k-fold cross validation
def train_data(splitted_dt, i):
    # extract train data
    train_dt = pd.concat([x for j,x in enumerate(splitted_dt) if j!=i])
    # divide into input and target
    train_input = train_dt.iloc[:, :input_size]
    train_target = train_dt.iloc[:, input_size:]
    # create Tensors to hold inputs and outputs
    X = torch.Tensor(train_input.values)
    Y = torch.Tensor(train_target.values)
    return X, Y


# get test data, used in k-fold cross validation
def test_data(splitted_dt, i):
    # extract test data
    test_dt = splitted_dt[i]
    # divide into input and target
    test_input = test_dt.iloc[:, :input_size]
    test_target = test_dt.iloc[:, input_size:]
    # create Tensors to hold inputs and outputs
    X = torch.Tensor(test_input.values)
    Y = torch.Tensor(test_target.values)
    return X, Y


# define regression model
class Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Regression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


# train the model, given X and Y
def train(X, Y, num_epochs, learning_rate, plot=True, to_print=False):
    # define regression model
    reg_model = Regression(input_size, output_size)
    # define optimiser
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=learning_rate)
    # store all losses for visualisation
    all_losses = []

    # start training
    for epoch in range(num_epochs):
        # perform forward pass: compute predicted y by passing x to the model
        Y_predicted = reg_model(X)

        # compute loss
        loss = loss_func(Y_predicted, Y)
        all_losses.append(loss.item())

        # print every 100 epochs
        if (epoch + 1) % 100 == 0 and to_print:
            print('Training Epoch: [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))

        # clear gradients before backward pass
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update parameters
        optimizer.step()

    # plot the accuracy of model on training data
    if plot:
        plt.figure()
        plt.plot(all_losses)
        plt.title('losses of model on training data')
        plt.show()

    # print confusion matrix and correct percentage
    Y_predicted = reg_model(X)
    loss = loss_func(Y_predicted, Y)
    total_num = Y_predicted.size(0)
    confusion = confusion_matrix(Y, Y_predicted)
    correctness = (100 * float(confusion[1])) / total_num
    if to_print:
        print('Confusion matrix for training:')
        print(confusion[0])
        print('Correct percentage in training data:', (100*float(confusion[1]))/total_num)

    return reg_model, loss, correctness


# perform testing on trained model, print test loss, confusion matrix and correctness
def test(X_test, Y_test, reg_model, to_print=False):
    Y_predicted_test = reg_model(X_test)
    test_loss = loss_func(Y_predicted_test, Y_test)
    if to_print:
        print('test loss: %f' % test_loss.item())
    total_num = Y_test.size(0)
    confusion = confusion_matrix(Y_predicted_test, Y_test)
    correctness = (100*float(confusion[1]))/total_num
    if to_print:
        print('Confusion matrix for testing:')
        print(confusion[0])
        print('Correct percentage in testing data:', correctness)
    return test_loss, correctness


# calculate confusion matrix, need to interpret output data into category
def confusion_matrix(Y, Y_predicted):
    confusion = torch.zeros(5, 5)
    correct_num = 0
    for i in range(Y.shape[0]):
        actual_class = interpret_output(Y[i])
        predicted_class = interpret_output(Y_predicted[i])
        confusion[actual_class[1]][predicted_class[1]] += 1
        if actual_class == predicted_class:
            correct_num += 1
    return confusion, correct_num


################################ main ###################################
if __name__ == "__main__":
    # initialize a genetic algorithm model
    ga = GA_model(DNA_size, pop_size, cross_rate, n_generations)

    highest_test_correctness = 0
    train_correctness = 0
    test_loss_best = 0
    train_loss_best = 0
    for t in range(run_times):
        # pre-process the data, using the function defined in preprocessing.py
        data = pre_process()

        # split data for later use (k cross validation)
        splitted_data = np.split(data, k_cross_validation)

        # train using cross validation
        all_train_losses = []
        all_test_losses = []
        all_train_correctness = []
        all_test_correctness = []
        for i in range(k_cross_validation):
            # extract train and test data, split input and target
            X_train, Y_train = train_data(splitted_data, i)
            X_test, Y_test = test_data(splitted_data, i)

            # train the model and print loss, confusion matrix and correctness
            reg_model, loss, correctness = train(X_train, Y_train, plot=False)

            # test the model on test data
            test_loss, test_correctness = test(X_test, Y_test, reg_model)

            # append losses and correctness
            all_train_losses.append(loss)
            all_test_losses.append(test_loss)
            all_train_correctness.append(correctness)
            all_test_correctness.append(test_correctness)

        # print average loss and correctness on training and testing data
        print("run number {}".format(str(t)))
        train_loss_avg = (sum(all_train_losses) / len(all_train_losses)).item()
        test_loss_avg = (sum(all_test_losses) / len(all_test_losses)).item()
        print('average loss on training data', train_loss_avg)
        print('average loss on testing data', test_loss_avg)
        train_correctness_avg = sum(all_train_correctness) / len(all_train_correctness)
        test_correctness_avg = sum(all_test_correctness) / len(all_test_correctness)
        print('average correctness on training data', train_correctness_avg)
        print('average correctness on testing data', test_correctness_avg)
        print('')

        # update highest
        if test_correctness_avg > highest_test_correctness:
            highest_test_correctness = test_correctness_avg
            train_correctness = train_correctness_avg
            test_loss_best = test_loss_avg
            train_loss_best = train_loss_avg

        # display performance of each model
        if plot_each_run:
            # losses
            plt.figure()
            plt.plot(all_train_losses, label='training data', color='blue')
            plt.plot(all_test_losses, label='testing data', color='red')
            plt.axhline(y=train_loss_avg, linestyle=':', label='training data average loss', color='blue')
            plt.axhline(y=test_loss_avg, linestyle=':', label='testing data average loss', color='red')
            plt.legend()
            plt.title('losses of model on training and testing data')
            plt.show()
            # correctness
            plt.figure()
            plt.plot(all_train_correctness, label='training data', color='blue')
            plt.plot(all_test_correctness, label='testing data', color='red')
            plt.axhline(y=train_correctness_avg, linestyle=':', label='training data average correctness', color='blue')
            plt.axhline(y=test_correctness_avg, linestyle=':', label='testing data average correctness', color='red')
            plt.legend()
            plt.title('correctness of model on training and testing data')
            plt.show()

    print("highest test correctness rate over 100 runs:", highest_test_correctness)
    print("corresponding training correctness rate:", train_correctness)
    print("corresponding testing loss:", test_loss_best)
    print("corresponding training loss:", train_loss_best)