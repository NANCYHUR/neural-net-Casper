"""
This script builds a neural network for classifying forest supra-type based on
dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process
import numpy as np


# step 1: load data and pre-process data
data = pre_process()

# hyper parameters
input_size = 20
hidden_size = 12
output_size = 4
num_epochs = 500
learning_rate = 0.01
k_cross_validation = 5

splitted_data = np.split(data, k_cross_validation)

