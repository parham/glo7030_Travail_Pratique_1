
""" 
    @name core file   
    @info   The core file includes all the implemented code that needs to be accessed across the other files.
    @organization: Laval University
    @author     Parham Nooralishahi
    @email      parham.nooralishahi.1@ulaval.ca
    @professor  Pascal Germain
    @semester   Winter 2021 
"""

from deeplib.datasets import load_mnist_v2
from torch.utils.data import Subset, DataLoader
from torchvision.datasets.mnist import MNIST
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

import imageio
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from poutyne import Callback, Dict

plt.rcParams['figure.dpi'] = 150

def save_model(model, file_path):
    """save_model saves the trained model in the specified path

    Args:
        model (nn.Module): the model to be saved.
        file_path (str): file path.
    """

    torch.save(model, file_path)

def load_model(file_path):
    """load_model loads the saved model from specified path.

    Args:
        file_path (str): file path to the trained model.

    Returns:
        nn.Module: the trained model.
    """
    return torch.load(file_path)

def load_datasets(split_percent=0.8):
    """load_datasets loads the dataset and split it into training, validation, and testing subsets based on the given split percentage.

    Args:
        split_percent (float, optional): The split ratio. Defaults to 0.8.

    Returns:
        train_dataset: the training subset of MNIST dataset,
        valid_dataset: the validation subset of MNIST dataset,
        testing_dataset: the testing subset of MNIST dataset
    """

    # Load MNIST
    print('Loading MNIST Dataset ...')
    training_ds, testing_dataset = load_mnist_v2()
    # Shuffle the data
    num_data = len(training_ds)
    print('Number of Loaded Data: %d' % num_data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split_len = math.floor(split_percent * num_data)
    # Training dataset
    train_indices = indices[:split_len]
    train_dataset = Subset(training_ds, train_indices)
    # Validation dataset
    valid_indices = indices[split_len:]
    valid_dataset = Subset(training_ds, valid_indices)

    return train_dataset, valid_dataset, testing_dataset
    
def load_datasets_q4():
    """load_datasets loads the dataset.

    Returns:
        train_dataset: the training subset of MNIST dataset,
        testing_dataset: the testing subset of MNIST dataset
    """

    # Load MNIST
    print('Loading MNIST Dataset ...')
    training_ds, testing_dataset = load_mnist_v2()
    # Shuffle the data
    num_data = len(training_ds)
    print('Number of Loaded Data: %d' % num_data)

    return training_ds, testing_dataset

class PhmMnist(nn.Module):
    """ PhmMnist is the designed nn model for learning MNIST dataset. """

    mnist_image_size = 28*28

    def __init__(self, num_classes=10):
        """ 
        Args:
            num_classes (int, optional): the number of classes need to be recognized. Defaults to 10.
        """

        super().__init__()
        self.layer_input = nn.Flatten()
        self.layer_h1 = nn.Linear(self.mnist_image_size, 256)
        self.layer_h2 = nn.Linear(256, 128)
        self.layer_h3 = nn.Linear(128, 64)
        self.layer_h4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer_input(x)
        x = F.relu(self.layer_h1(x))
        x = F.relu(self.layer_h2(x))
        x = F.relu(self.layer_h3(x))
        x = F.relu(self.layer_h4(x))
        return x

class PhmPlottingCallback(Callback):
    """ PhmPlottingCallback is the class containing the codes for visualizing the model's results. 
        The class also extends Callback class makes it able to attach to Poutyne experiment and register for the raised events.
    """

    def __init__(self):
        super().__init__()
        self.report = list()
        self.fig_list = list()
        self.fig, (self.loss_ax, self.acc_ax) = plt.subplots(2, 1)
        self.fig.tight_layout(pad=1.0)
        self.fig.suptitle('Statistics of training model')
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.ion()
        plt.show()

    def on_train_begin(self, logs: Dict):
        """ on_train_begin is a method invoked in the begining of each train. """
        pass

    def _plot(self, axis, epoch, train_data, valid_data, label, maxv=1.0, minv=0.0, stepv=0.05):
        axis.clear()
        plt.yticks(fontsize=11)
        axis.plot(epoch, train_data, '.-', label='Train')
        axis.plot(epoch, valid_data, '.-', label='Validation')
        axis.set_xlabel('epoch')
        axis.set_ylabel(label)
        # axis.set_yticks(np.arange(minv, maxv, stepv), minor = True)
        axis.set_yticks(np.arange(min(train_data + valid_data),
                                  max(train_data + valid_data), stepv), minor=True)
        axis.legend()
        axis.grid(True, which='both', axis='both', linestyle='--')

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        """ on_epoch_end is a method invoked in the end of each epoch. """

        self.report.append(logs)

        epoch = [item['epoch'] for item in self.report]
        train_loss = [item['loss'] for item in self.report]
        val_loss = [item['val_loss'] for item in self.report]
        train_acc = [item['acc'] for item in self.report]
        val_acc = [item['val_acc'] for item in self.report]

        canvas = FigureCanvasAgg(self.fig)
        # Loss
        self._plot(self.loss_ax, epoch, train_loss, val_loss, 'Loss')
        # Accuracy
        self._plot(self.acc_ax, epoch, train_acc,
                   val_acc, 'Accuracy', 100, 0, 5)

        plt.show()
        plt.pause(0.05)

        canvas.draw()
        fimg = np.array(canvas.renderer.buffer_rgba())
        self.fig_list.append(fimg)

    def on_train_end(self, logs: Dict):
        """ on_train_end is the method invoked in the end of each training. """
        
        if len(self.fig_list) > 0:
            imageio.mimsave("result.gif", self.fig_list, 'GIF')
            imageio.imwrite('result.png', self.fig_list[-1])

class DeadNeuronHook:
    """ DeadNeuronHook is a hook class designed to be used to measure the ratio of dead neurons.
        This class contains two hook method: (a) forward pass hook, (b) backward pass hook.
        The hooks are similar to event handlers for network's layers. 
        For instance, when a hook is registered for a layer for backward pass,
        this method will be invoked for each input data after the gradient is calculated. The same is correct
        for forward pass, which means the forward pass hook will be invoked for each given input after the output of the neuron
        for all the neurons in the layer is calcualted.
    """

    def __init__(self, name, thresh=0.99):
        self._name = name
        # dead_signals is a matrix with similar dimensions to the layer. The values shows the number of times the specific neuron shows 
        # the signs of being dead (zero gradient for backpass, zero output for forward pass).
        self.dead_signals = None
        # number_signals keep the total number of input samples.
        self.number_signals = 0
        # dead_thresh determines the ratio of tolerance to call a model perfect with 0% error rate.
        self.dead_thresh = thresh

    @property
    def name(self):
        """ layer's name """
        return self._name

    def calculate_dead_rate(self):
        """ calculate_dead_rate calculates the ratio of dead neurons for each layer.
        """

        # calculate the ratio of inactiveness for the neurons in the layer.
        # dead_signals is a matrix with similar dimensions to the layer. The values shows the number of times the specific neuron shows 
        # the signs of being dead (zero gradient for backpass, zero output for forward pass).
        # number_signals != 0 is checked to make sure if the calculate_dead_rate called by a user before any hook is called, the method
        # doesn't raise any exception.
        active_ratio = self.dead_signals / \
            self.number_signals if self.number_signals != 0 else 0
        dead_neurons = np.zeros(active_ratio.shape)
        # dead_neurons are detected by detecting the neurons that were inactive (or with zero gradient) for all the samples.
        dead_neurons[active_ratio >= self.dead_thresh] = 1
        # layer_ratio calculated by dividing the number of dead neurons to the total number of neurons (np.max(dead_neurons.shape)) 
        # in the layer.
        layer_ratio = (np.sum(dead_neurons) /
                       np.max(dead_neurons.shape)) * 100.0
        return layer_ratio, active_ratio, dead_neurons

    def hook_backward_dead_neuron(self, module, gradInput, gradOutput):
        """ hook_backward_dead_neuron is a hook method for backward pass call """

        grad = gradOutput[0].cpu().numpy()
        if self.dead_signals is None:
            self.dead_signals = np.zeros(grad.shape)

        tmp = np.zeros(grad.shape)
        tmp[grad == 0] = 1
        self.dead_signals += tmp
        self.number_signals += 1

    def hook_forward_dead_neuron(self, module, inputs, outputs):
        """ hook_backward_dead_neuron is a hook method for forward pass call """
        out = F.relu(outputs)
        out = out.detach().numpy()
        if self.dead_signals is None:
            self.dead_signals = np.zeros(out.shape)

        tmp = np.zeros(out.shape)
        tmp[out == 0] = 1
        self.dead_signals += tmp
        self.number_signals += 1

def calculate_dead_neoron_ratio(model, dataset, forward_check = False, thresh=0.99, batch_size=1):
    """ calculate_dead_neoron_ratio calculate the ratio of dead neurons for all layers.

    Args:
        model (nn.Module): the targeted model.
        dataset (Dataset): the loaded dataset.
        forward_check (bool, optional): determines whether it is a forward pass (True) or backward pass (False). Defaults to False.
        thresh (float, optional): the acceptable tolerance to the error in the model. Defaults to 0.99.
        batch_size (int, optional): the size of the batch. Defaults to 1.

    Returns:
        dict: a dictionary containing the ratio of dead neurons for all layers.
    """

    # Create the data loader
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2)

    # Loss Function Initialization
    loss_function = nn.CrossEntropyLoss()
    # Initialize the hooks for calculating dead neuron ratio
    layers = [module for module in model.modules() if type(module) == nn.Linear]
    stats = dict()
    hook_list = []
    # Create a hook (DeadNeuronHook) for tracking the dead neurons.
    for index in range(0, len(layers)):
        l = layers[index]
        name = 'layer %d' % index
        h = DeadNeuronHook(name, thresh=thresh)
        hn = None
        if forward_check:
            # Register forward pass hook.
            hn = l.register_forward_hook(h.hook_forward_dead_neuron)
        else:
            # Register backward pass hook.
            hn = l.register_backward_hook(h.hook_backward_dead_neuron)
        hook_list.append(hn)
        stats[name] = h

    model.train()
    # Feed the model with the given testing samples
    for inputs, targets in data_loader:
        # Reset the gradients
        model.zero_grad()
        # forward pass
        output = model(inputs)
        # Calculate the loss
        loss = loss_function(output, targets)
        # Calculate the gradients
        loss.backward()

    # Calculate the statistics
    res = dict()
    for key, hook in stats.items():
        # Calculate the ratio of dead neurons for the layers.
        layer_ratio, _, _ = hook.calculate_dead_rate()
        res[key] = layer_ratio

    # Remove the hook from the layers (Required for repeated  )
    for hn in hook_list:
        hn.remove()

    return res
