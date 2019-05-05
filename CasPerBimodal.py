"""
This script builds a neural network with CasPer and Bimodal techniques
for classifying forest supra-type based on dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process, interpret_output
from NN import confusion_matrix, train_data, test_data
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

run_time = 100

# hyper parameters
input_size = 20
output_size = 4
num_neurons = 14
learning_rate_1 = 0.2
learning_rate_2 = 0.005
learning_rate_3 = 0.001
k_cross_validation = 5
p_value = 4

# parameters for Bimodal Distribution Removal
variance_thresh = 0.00001
variance_thresh_halting = 0.000001
alpha = 0.5

# define loss function
loss_func = nn.MSELoss(reduction='none')


# define regression model
class Regression(nn.Module):
    def __init__(self, input_size, output_size, all_losses, num_neurons, p_value):
        super(Regression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = 0
        self.losses = all_losses
        self.num_neurons = num_neurons
        self.input_to_output = nn.Linear(input_size, output_size)
        self.p = p_value
        # all_to_output: a list containing all Linears from all neurons to output neurons
        self.all_to_output = []
        # all_to_hidden: a list containing all Linears from previous neurons to a hidden neuron
        self.all_to_hidden = []

        # define optimizer
        self.optimizers = []
        optimizer = torch.optim.Rprop(self.parameters(), lr=learning_rate_1)
        self.optimizers.append(optimizer)

    def forward(self, x):   # size of x: (152, 20)
        if self.n_hidden == 0 or len(self.losses) < 2:
            self.install = True
        elif self.current_neuron_epoch<15+self.p*self.n_hidden:
            self.install = False
        else:
            loss1 = self.losses[-1]
            loss2 = self.losses[-2]
            rms1 = math.sqrt(loss1)
            rms2 = math.sqrt(loss2)
            decrease = (rms2 - rms1) / rms2
            self.install = 0 < decrease <= 0.01

        if self.install:
            if self.n_hidden == self.num_neurons:
                self.n_hidden -= 1
                return None
            all_to_new = nn.Linear(self.input_size+self.n_hidden, 1)
            self.all_to_hidden.append(all_to_new)
            new_to_output = nn.Linear(1, self.output_size, bias=False)
            self.all_to_output.append(new_to_output)

            # change learning rate of optimizers
            if len(self.optimizers) == 1:
                optim = self.optimizers[0]
                for g in optim.param_groups:
                    g['lr'] = learning_rate_3
            else:
                optim = self.optimizers[-2]
                for g in optim.param_groups:
                    g['lr'] = learning_rate_3
                optim = self.optimizers[-1]
                for g in optim.param_groups:
                    g['lr'] = learning_rate_3
            self.optimizers.append(torch.optim.Rprop([new_to_output.weight], lr=learning_rate_2))
            self.optimizers.append(torch.optim.Rprop([all_to_new.weight, all_to_new.bias], lr=learning_rate_1))

            self.n_hidden += 1
            self.current_neuron_epoch = 0

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

        self.current_neuron_epoch += 1

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
def train(X, Y, plot=False, to_print=False):
    # store all losses for visualisation
    all_losses = []

    # define regression model
    reg_model = Regression(input_size, output_size, all_losses, num_neurons, p_value)

    # start training
    while True:
        # perform forward pass: compute predicted y by passing x to the model
        Y_predicted = reg_model(X)

        if Y_predicted is None:
            break

        # compute loss
        loss = loss_func(Y_predicted, Y)
        loss_avg = torch.mean(loss)
        all_losses.append(loss_avg.item())

        # print every neuron
        if reg_model.install and reg_model.n_hidden != 1 and to_print:
            print('Training Neuron: [%d/%d], Loss: %.4f' % (reg_model.n_hidden, num_neurons, all_losses[-2]))

            losses = torch.mean(loss, dim=1)
            # plt.figure()
            # plt.hist(x=losses.detach().numpy(), bins='auto')
            # plt.show()
            variance = torch.var(losses)
            if variance.item() < variance_thresh_halting:
                break  # halt to avoid overfitting
            std = torch.std(loss)
            error_thresh = loss_avg + std.item() * alpha
            survived_indices = [i for i in range(len(loss)) if losses[i].item() < error_thresh]
            indices = torch.LongTensor(survived_indices)
            X = X[indices]
            Y = Y[indices]

        # clear gradients before backward pass
        reg_model.clear_grad()
        # perform backward pass
        loss_avg.backward()
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
    loss_avg = torch.mean(loss).item()
    total_num = Y_predicted.size(0)
    confusion = confusion_matrix(Y, Y_predicted)
    correctness = (100 * float(confusion[1])) / total_num
    if to_print:
        print('Confusion matrix for training:')
        print(confusion[0])
        print('Correct percentage in training data:', (100 * float(confusion[1])) / total_num)

    return reg_model, loss_avg, correctness


# perform testing on trained model, print test loss, confusion matrix and correctness
def test(X_test, Y_test, reg_model, to_print=False):
    Y_predicted_test = reg_model(X_test)
    test_loss = loss_func(Y_predicted_test, Y_test)
    test_loss_avg = torch.mean(test_loss).item()
    if to_print:
        print('test loss: %f' % test_loss.item())
    total_num = Y_test.size(0)
    confusion = confusion_matrix(Y_predicted_test, Y_test)
    correctness = (100 * float(confusion[1])) / total_num
    if to_print:
        print('Confusion matrix for testing:')
        print(confusion[0])
        print('Correct percentage in testing data:', correctness)
    return test_loss_avg, correctness


################################ main ###################################
highest_test_correctness = 0
train_correctness = 0
test_loss_best = 0
train_loss_best = 0
for r in range(run_time):
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
        reg_model, loss, correctness = train(X_train, Y_train)

        # test the model on test data
        test_loss, test_correctness = test(X_test, Y_test, reg_model)

        # append losses and correctness
        all_train_losses.append(loss)
        all_test_losses.append(test_loss)
        all_train_correctness.append(correctness)
        all_test_correctness.append(test_correctness)

    # print average loss and correctness on training and testing data
    train_loss_avg = (sum(all_train_losses) / len(all_train_losses))
    test_loss_avg = (sum(all_test_losses) / len(all_test_losses))
    print('average loss on training data', train_loss_avg)
    print('average loss on testing data', test_loss_avg)
    train_correctness_avg = sum(all_train_correctness) / len(all_train_correctness)
    test_correctness_avg = sum(all_test_correctness) / len(all_test_correctness)
    print('average correctness on training data', train_correctness_avg)
    print('average correctness on testing data', test_correctness_avg)

    # update highest
    if test_correctness_avg > highest_test_correctness:
        highest_test_correctness = test_correctness_avg
        train_correctness = train_correctness_avg
        test_loss_best = test_loss_avg
        train_loss_best = train_loss_avg

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

print("highest test correctness rate over 100 runs:", highest_test_correctness)
print("corresponding training correctness rate:", train_correctness)
print("corresponding testing loss:", test_loss_best)
print("corresponding training loss:", train_loss_best)