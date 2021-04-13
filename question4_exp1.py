
""" 
    @name question #4   
    @info   In a code file named question4.py, develop a neural network architecture using
            only fully connected layers with ReLU activation function, and obtain an error rate of 0% on
            the MNIST training set. To load the dataset, use the load_mnist function from the deeplib
            library provided with the lab ; use the train and test split sets provided by this function.
    
    @organization: Laval University
    @author     Parham Nooralishahi
    @email      parham.nooralishahi.1@ulaval.ca
    @professor  Pascal Germain
    @semester   Winter 2021 
"""

import os
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from poutyne import CSVLogger, Experiment


#############################
### Load developed codes  ###
from parham_core import PhmMnist, PhmPlottingCallback, load_datasets, load_datasets_q4, save_model
#############################

# Information
from prettytable import PrettyTable

table = PrettyTable(['Information', ' '])
table.add_row(['Question', '#4'])
table.add_row(['Organization', 'Laval University'])
table.add_row(['Author', 'Parham Nooralishahi'])
table.add_row(['Email', 'parham.nooralishahi.1@ulaval.ca'])
table.add_row(['Professor', 'Pascal Germain'])
table.add_row(['Semester', 'Winter 2021'])
print(table)


####### PyTorch Initialization 
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
print('Learning device : %s' % ('CPU' if cuda_device == 0 else 'GPU'))
##############################

# Hyperparameters
batch_size = 32
learning_rate = 0.2
num_epochs = 60
num_classes = 10

# Hyperparameters
print()
table = PrettyTable(['Hyperparameters', 'Value'])
table.add_row(['Batch Size', batch_size])
table.add_row(['Learning Rate', learning_rate])
table.add_row(['Number of Epoch', num_epochs])
table.add_row(['Number of Classes', num_classes])
print(table)


# Initialize Data loaders
train_dataset, test_dataset = load_datasets_q4()
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=2)

# Model Initialization
model = PhmMnist()

print('\n=====\tDefined Model\t=====')
print(model)
print('=========================\n')
# Optimizer Initialization
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss Function Initialization
loss_function = nn.CrossEntropyLoss()
# Train the model
save_path = './'

callbacks = [
    PhmPlottingCallback(),
    # Save the losses and accuracies for each epoch in a TSV.
    CSVLogger(os.path.join(save_path, 'log.tsv'), separator='\t'),
]

# Create & Initialize Experiment
exp_q4 = Experiment(
    './results/question_4', model,
    device=device,
    optimizer=optimizer,
    loss_function=loss_function,
    batch_metrics=['accuracy'],
    epoch_metrics=['f1'],
    task='classif'
)

# Use Poutyne experiment to train the model.
exp_q4.train(train_generator=train_loader, valid_generator=test_loader,
             epochs=num_epochs, callbacks=callbacks)

print(exp_q4.test(test_loader))

# Save the trained model
save_model(model, './model.phm')