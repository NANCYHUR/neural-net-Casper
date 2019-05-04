"""
This script builds a neural network with CasPer technique
for classifying forest supra-type based on dataset: GIS.

The dataset is naturally a classification problem, however, we modify it into
a regression problem by equilateral coding, to avoid difficult learning.
"""

from preprocessing import pre_process, interpret_output
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

