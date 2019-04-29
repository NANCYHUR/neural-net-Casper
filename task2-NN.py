"""
This script builds a neural network for classifying forest supra-type based on
dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process, interpret_output
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# hyper parameters
input_size = 20
hidden_size = 12
output_size = 4
num_epochs = 500
learning_rate = 0.01
k_cross_validation = 5

# define loss function
loss_func = nn.BCELoss()


# get train data, used in k-fold cross validation
def train_data(splitted_dt, i):
    # extract train data
    train_dt = pd.concat([x for j,x in enumerate(splitted_dt) if j!=i])
    # divide into input and target
    train_input = train_dt.iloc[:, :input_size]
    train_target = train_dt.iloc[:, input_size:]
    # create Tensors to hold inputs and outputs
    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).float()
    return X, Y


# get test data, used in k-fold cross validation
def test_data(splitted_dt, i):
    # extract test data
    test_dt = splitted_dt[i]
    # divide into input and target
    test_input = test_dt.iloc[:, :input_size]
    test_target = test_dt.iloc[:, input_size:]
    # create Tensors to hold inputs and outputs
    X = torch.Tensor(test_input.values).float()
    Y = torch.Tensor(test_target.values).float()
    return X, Y


# define regression model
class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


# train the model, given X and Y
def train(X, Y, plot=True):
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
        # Y_predicted = Y_predicted.view(len(Y_predicted))
        # compute loss
        loss = loss_func(Y_predicted, Y)
        all_losses.append(loss.item())

        # print every 100 epochs
        if (epoch + 1) % 100 == 0:
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
        plt.title('accuracy of model on training data')
        plt.show()

    # print confusion matrix and correct percentage
    Y_predicted = reg_model(X)
    loss = loss_func(Y_predicted, Y)
    total_num = Y_predicted.size(0)
    confusion = confusion_matrix(Y, Y_predicted)
    print('Confusion matrix for training:')
    print(confusion[0])
    print('Correct percentage in training data:', (100*float(confusion[1]))/total_num)

    return reg_model, loss


# perform testing on trained model, print test loss, confusion matrix and correctness
def test(X_test, Y_test, reg_model):
    Y_predicted_test = reg_model(X_test)
    test_loss = loss_func(Y_predicted_test, Y_test)
    print('test loss: %f' % test_loss.item())
    total_num = Y_test.size(0)
    confusion = confusion_matrix(Y_predicted_test, Y_test)
    print('Confusion matrix for testing:')
    print(confusion[0])
    print('Correct percentage in testing data:', (100*float(confusion[1]))/total_num)
    return test_loss


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
# pre-process the data, using the function defined in preprocessing.py
data = pre_process()

# split data for later use (k cross validation)
splitted_data = np.split(data, k_cross_validation)

# turn plot on TODO: something wrong, there is no plot
plt.ion()

# train using cross validation
all_losses = []
for i in range(k_cross_validation):
    # extract train and test data, split input and target
    X_train, Y_train = train_data(splitted_data, i)
    X_test, Y_test = test_data(splitted_data, i)

    # train the model and print loss, confusion matrix and correctness
    reg_model, loss = train(X_train, Y_train)

    # test the model on test data
    test_loss = test(X_test, Y_test, reg_model)

    all_losses.append((loss, test_loss))

# TODO: grab the lowest loss, stick to that model (print result)j
