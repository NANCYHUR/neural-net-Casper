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
num_epochs = 500
learning_rate_1 = 0.2
learning_rate_2 = 0.005
learning_rate_3 = 0.001
k_cross_validation = 5

# define loss function
loss_func = nn.BCEWithLogitsLoss()


# define SARprop
class SARprop(torch.optim.Rprop):
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        super(SARprop, self).__init__()

    ############# most of the following code is modified from torch.optim.rprop #############
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Rprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev'] = torch.zeros_like(p.data)
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])

                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']
                step_size = state['step_size']

                state['step'] += 1

                sign = grad.mul(state['prev']).sign()
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

                # update stepsizes with step size updates
                step_size.mul_(sign).clamp_(step_size_min, step_size_max)
                step_size -=

                # for dir<0, dfdx=0
                # for dir>=0 dfdx=dfdx
                grad = grad.clone()
                grad[sign.eq(etaminus)] = 0

                # update parameters
                p.data.addcmul_(-1, grad.sign(), step_size)

                state['prev'].copy_(grad)

        return loss
    ############################################################################################

# define regression model
class Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Regression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = 0
        self.epoch = 0
        self.input_to_output = nn.Linear(input_size, output_size)
        # all_to_output: a list containing all Linears from all neurons to output neurons
        self.all_to_output = []
        # all_to_hidden: a list containing all Linears from previous neurons to a hidden neuron
        self.all_to_hidden = []

        # define optimizer TODO modified Rprop
        self.optimizers = []
        optimizer = torch.optim.Rprop(self.parameters(), lr=learning_rate_1)
        self.optimizers.append(optimizer)

    def forward(self, x):   # size of x: (152, 20)
        install = False
        if self.n_hidden == 0:
            install = True
        else:  # TODO https://piazza.com/class/jsjthqq2z655mg?cid=177
            pass

        if install:
            all_to_new = nn.Linear(self.input_size+self.n_hidden, 1)
            self.all_to_hidden.append(all_to_new)
            new_to_output = nn.Linear(1, self.output_size, bias=False)
            self.all_to_output.append(new_to_output)

            # change learning rate of optimizers
            if len(self.optimizers) == 1:
                self.optimizers[0] = torch.optim.Rprop(self.parameters(), lr=learning_rate_3)
            else:
                self.optimizers[-2] = torch.optim.Rprop(self.optimizers[-2].param_groups, lr=learning_rate_3)
                self.optimizers[-1] = torch.optim.Rprop(self.optimizers[-1].param_groups, lr=learning_rate_3)
            self.optimizers.append(torch.optim.Rprop([new_to_output.weight], lr=learning_rate_2))
            self.optimizers.append(torch.optim.Rprop([all_to_new.weight, all_to_new.bias], lr=learning_rate_1))

            self.n_hidden += 1
            self.last_added = self.epoch

        # perform forward propagation from all previous neurons to each hidden neuron
        neurons_all = x
        for i in range(self.n_hidden):
            linear = self.all_to_hidden[i]
            neuron = linear(neurons_all[:, :i+self.input_size])
            neurons_all = torch.cat((neurons_all, neuron), 1)

        # perform forward propagation from all neurons to output neurons
        logic = self.input_to_output(x)
        for i in range(self.n_hidden):
            logic = logic + self.all_to_output[i](neurons_all[:, i+self.input_size].unsqueeze(1))
        out = torch.sigmoid(logic)
        # print(self.n_hidden)

        self.epoch += 1
        return out

    # clear gradients, used before backward
    def clear_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    # this is used for updating parameters
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


# train the model, given X and Y
def train(X, Y, plot=True, to_print=False):
    # define regression model
    reg_model = Regression(input_size, output_size)
    # define optimiser

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
        reg_model.clear_grad()
        # perform backward pass
        loss.backward()
        # update parameters
        reg_model.step()

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