
""" 
    @name question #5b  
    @info   In a code file named question5b.py.
    
    @organization: Laval University
    @author     Parham Nooralishahi
    @email      parham.nooralishahi.1@ulaval.ca
    @professor  Pascal Germain
    @semester   Winter 2021 
"""

from prettytable import PrettyTable
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.init as ninit
import json

from poutyne import ModelCheckpoint, CSVLogger, Experiment

#############################
### Load developed codes  ###
import parham_core as core
#############################

def normal_init(layer):
    if isinstance(layer, nn.Linear):
        ninit.normal_(layer.weight, 0.5, 0.5)
        if layer.bias is not None:
            ninit.normal_(layer.bias, 0.5, 0.5)

def uniform_init(layer):
    if isinstance(layer, nn.Linear):
        ninit.uniform_(layer.weight, -1.0, 1.0)
        if layer.bias is not None:
            ninit.uniform_(layer.bias, -1.0, 1.0)

def constant_init(layer):
    if isinstance(layer, nn.Linear):
        ninit.ones_(layer.weight) * 0.1
        if layer.bias is not None:
            ninit.ones_(layer.bias) * 0.1

def kaiming_uniform(layer):
    if isinstance(layer, nn.Linear):
        ninit.kaiming_uniform_(layer.weight, a=1.0, mode='fan_in')
        if layer.bias is not None:
            ninit.zeros_(layer.bias)

def xavier_normal(layer):
    if isinstance(layer, nn.Linear):
        ninit.xavier_normal_(layer.weight, gain=1.0)
        if layer.bias is not None:
            ninit.zeros_(layer.bias)

def create_and_train(name, init_method=normal_init, learning_rate=0.2, num_epochs=50):
    """ create_and_train is a method to create subsets from loaded datast, calculate the ratio of dead neurons, and train and test the model.

    Args:
        name (str): a selected name to save the results.
        init_method (func, optional): The strategy for weight and bias strategy. Defaults to normal_init.
        learning_rate (float, optional): The learning rate for training process. Defaults to 0.2.
        num_epochs (int, optional): The number of epochs. Defaults to 50.

    Returns:
        dict: a dictionary containing all the results
    """
    
    train_dataset, valid_dataset, test_dataset = core.load_datasets()

    indices = list(range(256))
    test_ds = Subset(test_dataset, indices)

    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=2, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=32, num_workers=2, shuffle=True)

    # Model Initialization
    model = core.PhmMnist()
    model.to(torch.device("cpu"))

    print('\n=====\tDefined Model\t=====')
    print(model)
    print('=========================\n')
    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Loss Function Initialization
    loss_function = nn.CrossEntropyLoss()
    # Weight Initialization
    model.apply(init_method)

    # Before training dead neuron
    indices = list(range(256))
    test_ds = Subset(test_dataset, indices)

    res_before = core.calculate_dead_neoron_ratio(model, test_ds, forward_check = False, thresh=1.0)

    # Create & Initialize Experiment
    exp_q5b = Experiment(
        './results/question_5b_' + name, model,
        device=device,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_metrics=['accuracy'],
        task='classif'
    )

    # Use Poutyne experiment to train the model.
    model.train()
    res_train_acc = exp_q5b.train(train_generator=train_loader, valid_generator=valid_loader, epochs=num_epochs)
    
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=2)
    res_test_acc = exp_q5b.test(test_loader)

    res_after = core.calculate_dead_neoron_ratio(model, test_ds, forward_check = False, thresh=1.0)

    rest = dict()
    rest['dead_neuron_before'] = res_before
    rest['training'] = res_train_acc
    rest['testing'] = res_test_acc
    rest['dead_neuron_after'] = res_after

    # Write the results into the file
    with open('./results/' + name + '.json', 'w') as f:
        json.dump(rest, f, sort_keys=True, indent=4)

    torch.save(model, './results/' + 'model_' + name + '.phm')

    return rest


# Information
table = PrettyTable(['Information', ' '])
table.add_row(['Question', '#4'])
table.add_row(['Organization', 'Laval University'])
table.add_row(['Author', 'Parham Nooralishahi'])
table.add_row(['Email', 'parham.nooralishahi.1@ulaval.ca'])
table.add_row(['Professor', 'Pascal Germain'])
table.add_row(['Semester', 'Winter 2021'])
print(table)

# PyTorch Initialization
cuda_device = 0
device = torch.device("cpu")
print('Learning device : %s' % ('CPU' if cuda_device == 0 else 'GPU'))
##############################

# Hyperparameters
batch_size = 32
learning_rate = 0.2
num_epochs = 55
num_classes = 10

# Hyperparameters
print()
table = PrettyTable(['Hyperparameters', 'Value'])
table.add_row(['Batch Size', batch_size])
table.add_row(['Learning Rate', learning_rate])
table.add_row(['Number of Epoch', num_epochs])
table.add_row(['Number of Classes', num_classes])
print(table)

print(create_and_train('normal_init', init_method=normal_init, num_epochs=num_epochs))
print(create_and_train('uniform_init', init_method=uniform_init, num_epochs=num_epochs))
print(create_and_train('constant_init', init_method=constant_init, num_epochs=num_epochs))
print(create_and_train('xavier_normal', init_method=xavier_normal, num_epochs=num_epochs))
print(create_and_train('kaiming_uniform', init_method=kaiming_uniform, num_epochs=num_epochs))
