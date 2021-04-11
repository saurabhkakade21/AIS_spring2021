# Imports
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, utils
from torchvision.transforms import ToTensor, Lambda, Resize, Compose, transforms
import matplotlib.pyplot as plt

# initializing SummaryWriter instances for subtrain and validation
writer_dict = {}

for x in 'convolutional', 'fullyConnected':
    for y in "subtrain", 'validation':
        writer_dict[x + '_' + y] = SummaryWriter("runs/" + y + ' ' + 'with' + ' ' + x)

def getBestEpochs(subtrainResults):
    minimumLossValue = min(subtrainResults.values())
    for keyValue in subtrainResults:
        if subtrainResults[keyValue] == minimumLossValue:
            minimumLossValueKey = keyValue

    return minimumLossValueKey

# Fully connected deep neural network
class FullyConnectedNetwork(torch.nn.Module):

    def __init__(self, input_size, h1, h2, output_size):
        super().__init__() 
        self.layer_1 = torch.nn.Linear(input_size, h1)
        self.layer_2 = torch.nn.Linear(h1, h2)
        self.layer_3 = torch.nn.Linear(h2, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# LeNet Convolution neural network
class LeNet(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels= 6, kernel_size=5)
      self.conv2 = torch.nn.Conv2d(6, 16, 5)
      self.fc1 = torch.nn.Linear(4*4*16, 120)
      self.fc2 = torch.nn.Linear(120, 84)
      self.output = torch.nn.Linear(84, 10)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*16)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.output(x)
      return x

NUM_SPLITS = 3

full_data_dict = {}

for data_name in ["FashionMNIST", "KMNIST"]:
    # Downloading the datasets
    data_loader = getattr(datasets, data_name)

    full_data_dict[data_name] = data_loader(root="data", train=False, download=True, transform=ToTensor())

    for splitNum in range(NUM_SPLITS):
        # Splitting the data thrice
        validationDataset, subtrainDataset = torch.utils.data.random_split(full_data_dict[data_name], (3333, 6667))

        # Initializing a dataloader for subtrain and validation data which will be used to compute the loss in each epoch with respect to all examples in the subtrain/validation sets
        trainDataLoader = torch.utils.data.DataLoader(subtrainDataset, batch_size=32, shuffle=True)
        validationDataLoader = torch.utils.data.DataLoader(validationDataset, batch_size=32, shuffle=True)

        # print("Length of training data: ", len(subtrainDataset))
        # print("Length of validation data: ", len(validationDataset))

        # Making an instance of the loss function
        torch.nn.CrossEntropyLoss()

        # Initialization
        convolutional = LeNet()

        fullyConnected = FullyConnectedNetwork(784, 300, 100, 10)

        dataLoader = {'subtrain': trainDataLoader, 'validation': validationDataLoader}

        neuralNetworks = {'convolutional':convolutional, 'fullyConnected':fullyConnected}

        for model in "fullyConnected", "convolutional":
            # Instantiating an optimizer
            optimizer = optim.Adam(neuralNetworks[model].parameters(),lr = 0.001)

            # Instance of loss function
            criterion = torch.nn.CrossEntropyLoss()

            epochs = 10

            subtrainResults = {}

            for epoch in range(epochs):

                lossDict = {"epoch":epoch, "model":model}

                for s in "subtrain", "validation":

                    if s == "subtrain":

                        trainLoss = 0

                        for bIndex, (data,target) in enumerate(dataLoader[s]):
                            
                            pred = neuralNetworks[model](data) #predictions
                            loss = criterion(pred, target) #loss
                            optimizer.zero_grad() #zero the gradient
                            loss.backward() #compute gradient
                            optimizer.step() #Update the neural network weights using gradient

                            trainLoss += loss.item()

                        # Computing the loss
                        trainLoss /= len(dataLoader[s])
                        loss = trainLoss
                        writer_dict[model + "_" + s].add_scalar('Train/Loss',loss,epoch)

                    else:
                        trainLoss = 0
                        for data, target in dataLoader[s]:
                            pred = neuralNetworks[model](data)
                            trainLoss += criterion(pred,target).item() 

                        # Computing the loss
                        trainLoss /= len(dataLoader[s])
                        loss = trainLoss
                        subtrainResults[epoch] = loss
                        writer_dict[model + "_" + s].add_scalar('Val/Loss',loss,epoch)

                    lossDict[s] = loss

            # Printing the values
            selectedEpochs = getBestEpochs(subtrainResults)     
            print("DATA=%s model=%s testFold=%d selectedEpochs=%d" % (data_name, model, splitNum, selectedEpochs))