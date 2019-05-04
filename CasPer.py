"""
This script builds a neural network with CasPer technique
for classifying forest supra-type based on dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process, interpret_output
from NN import confusion_matrix, train_data, test_data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# hyper parameters
input_size = 20
output_size = 4
num_epochs = 1000
learning_rate = 0.2
k_cross_validation = 5

# define loss function
loss_func = nn.BCEWithLogitsLoss()


class My_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        self.has_bias = bias

    # this feature append the weights of 'other_linear' to the end of self
    # keep the bias unchanged
    def update(self, other_linear):
        self.weight = nn.parameter.Parameter(torch.cat((self.weight, other_linear.weight), 1))


# define regression model
class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = 0
        # all_to_output: the weights from all neurons to output neurons
        self.all_to_output = My_Linear(input_size, output_size)
        # all_to_hidden: a list containing all Linears from previous neurons to a hidden neuron
        self.all_to_hidden = []

    def forward(self, x):   # size of x: (152, 20)
        install = False
        if self.n_hidden == 0:
            install = True
        else:  # https://piazza.com/class/jsjthqq2z655mg?cid=177
            pass

        if install:
            all_to_new = nn.Linear(self.input_size+self.n_hidden, 1)
            self.all_to_hidden.append(all_to_new)
            new_to_output = nn.Linear(1, self.output_size)
            self.all_to_output.update(new_to_output)

            self.n_hidden += 1

        # perform forward propagation from all previous neurons to each hidden neuron
        neurons_val = torch.zeros((x.size(0), self.n_hidden))
        neurons_val = torch.cat((x, neurons_val), 1)
        for i in range(self.n_hidden):
            linear = self.all_to_hidden[i]
            neurons_val[:, i+self.input_size] = torch.squeeze(linear(neurons_val[:, :i+self.input_size]))

        # perform forward propagation from all neurons to output neurons
        print(self.all_to_output.weight.size())
        print(neurons_val.size())
        out = self.all_to_output(neurons_val)
        return out


# train the model, given X and Y
def train(X, Y, plot=True, to_print=False):
    # define regression model
    reg_model = Regression(input_size, output_size)
    # define optimiser
    optimizer = torch.optim.Rprop(reg_model.parameters(), lr=learning_rate)
    # store all losses for visualisation
    all_losses = []

    # start training
    for epoch in range(num_epochs):
        # perform forward pass: compute predicted y by passing x to the model
        Y_predicted = reg_model(X)
        # Y_predicted = Y_predicted.view(len(Y_predicted))
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


################################ main ###################################
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
train_loss_avg = (sum(all_train_losses) / len(all_train_losses)).item()
test_loss_avg = (sum(all_test_losses) / len(all_test_losses)).item()
print('average loss on training data', train_loss_avg)
print('average loss on testing data', test_loss_avg)
train_correctness_avg = sum(all_train_correctness) / len(all_train_correctness)
test_correctness_avg = sum(all_test_correctness) / len(all_test_correctness)
print('average correctness on training data', train_correctness_avg)
print('average correctness on testing data', test_correctness_avg)

# display performance of each model
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