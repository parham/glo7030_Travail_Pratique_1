

""" 
    @name question #5a
    @info   In a code file named question5a.py, write a function to make statistics on dead neurons of a
    neural network with ReLU activations.
    @organization: Laval University
    @author     Parham Nooralishahi
    @email      parham.nooralishahi.1@ulaval.ca
    @professor  Pascal Germain
    @semester   Winter 2021 
"""

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Subset

import matplotlib.pyplot as plt

#############################
### Load developed codes  ###
import parham_core as core
#############################

# Information
from prettytable import PrettyTable

table = PrettyTable(['Information', ' '])
table.add_row(['Question', '#5a'])
table.add_row(['Organization', 'Laval University'])
table.add_row(['Author', 'Parham Nooralishahi'])
table.add_row(['Email', 'parham.nooralishahi.1@ulaval.ca'])
table.add_row(['Professor', 'Pascal Germain'])
table.add_row(['Semester', 'Winter 2021'])
print(table)


if __name__ == '__main__':
    # The path to the trained model.
    model_path = './results/model_q4_exp2.phm' # The model for experiment 1 is ./results/model_q4_exp1.phm 
    device = torch.device("cpu")
    # Load the model
    model = core.load_model(model_path)
    print(model)
    model.to(device)
    # Load the dataset and prepare the subsets for training, validation, and testing.
    train_dataset, valid_dataset, testing_dataset = core.load_datasets()
    # Calculate the ratio of dead neurons for each layer.
    res = core.calculate_dead_neoron_ratio(model, testing_dataset, forward_check = False, thresh=1.0)
    # The method returns a dictionary containing the layers and the ratio of dead neurons
    keys = []
    vals = []
    for k, s in res.items():
        print(k + ' : ' + str(s))
        keys.append(k)
        vals.append(s)

    x_pos = [i for i, _ in enumerate(keys)]
    plt.bar(x_pos, vals, color='green')
    plt.xlabel("")
    plt.ylabel("Dead Neuron Ratio (0-100%)")
    plt.title("Layers")
    plt.xticks(x_pos, keys)
    plt.show()
