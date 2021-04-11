
# Advanced Intelligence Systems: Assignment 03 by Saurabh Kakade (sk2354@nau.edu)

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, utils, transforms
from torchvision.transforms import ToTensor, Lambda, Resize, Compose, transforms
import matplotlib.pyplot as plt

#*******************************************************************

class FullyConnected(torch.nn.Module):

    def __init__(self, input_size=784, h1=300, h2=100, output_size=10):
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

#*******************************************************************
    
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
      # use x.shape to check the current size
      # print (x.shape)
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*16)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.output(x)
      print(x)
      return x

#*******************************************************************

# transform using Compose, Resize, ToTensor from torchvision.transforms, 
#which will convert each input into a 28x28 Tensor.

transform_data = transforms.Compose([transforms.ToTensor(), transforms.Resize((28,28))])

#*******************************************************************

#download the Fashion MNIST train data

train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_data)

#*******************************************************************

#randomly split the train data

train_dataset, val_dataset = random_split(train_set,[50000,10000])

print(train_dataset[0])

#*******************************************************************

#Make a torch.utils.data.DataLoader for each set 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

print("Length of Train data: {}".format(len(train_loader.dataset)))

print("Length of Test data: {}".format(len(validation_loader.dataset)))

#*******************************************************************

#Initialization of two neural network: LeNet and Fully Connected network

torch.nn.CrossEntropyLoss()

lenet_model = LeNet()

print(lenet_model)

fully_connected_model = FullyConnected()

print(fully_connected_model)

#*******************************************************************

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initializing two SummaryWriter instances, one for subtrain, one for validation.

writer_dict = {}

for mm in 'LeNet', 'fully_connected_model':

    for ss in "subtrain", 'validation':

        writer_dict[mm + '_' + ss] = SummaryWriter("runs/" + ss + '_' + 'with' + '_' + mm)

#*******************************************************************


d_loader = {'subtrain': train_loader, 'validation': validation_loader}

neural_networks = {'LeNet':lenet_model, 'fully_connected_model':fully_connected_model}


#*******************************************************************

#use a for loop over the two networks

for pattern in "LeNet", "fully_connected_model":

    #For each network, begin by instantiating an optimizer

    optimizer = optim.Adam(neural_networks[pattern].parameters(),lr = 0.001)

    #Make an instance of the loss function

    criterion = torch.nn.CrossEntropyLoss()

    #use a for loop over epochs of learning

    max_epochs = 1

    #*******************************************************************

    for epoch in range(max_epochs):

        loss_dict = {"epoch":epoch, "pattern":pattern}

        for s in "subtrain", "validation":

            #*******************************************************************

            if s == "subtrain":
                train_loss = 0

                #writing a for loop over batches using your DataLoader (for x,y in data_loader).
                for batch_index, (data,target) in enumerate(d_loader[s]):
                    
                    pred = neural_networks[pattern](data) #predictions
                    loss = criterion(pred, target) #loss
                    optimizer.zero_grad() #zero the gradient
                    loss.backward() #compute gradient
                    optimizer.step() #Update the neural network weights using gradient

                    train_loss += loss.item()


                #Compute loss
                train_loss /= len(d_loader[s])
                loss = train_loss
                writer_dict[pattern + "_" + s].add_scalar('Train/Loss',loss,epoch)

                # print("Subtrain: {}".format(pred))
                # print(max(pred))

            else:
                train_loss = 0
                for data,target in d_loader[s]:
                    pred = neural_networks[pattern](data)
                    train_loss += criterion(pred,target).item() 

                   
                #Compute loss
                train_loss /= len(d_loader[s])
                loss = train_loss
                writer_dict[pattern + "_" + s].add_scalar('Val/Loss',loss,epoch)

                # print("Val: {}".format(pred))
                # print(max(pred))


            loss_dict[s] = loss

        #*******************************************************************

        #print these values on the screen and log these loss values to the tensorboard writer     
        print("pattern=%(pattern)s epoch=%(epoch)d subtrain=%(subtrain)f validation=%(validation)f" % loss_dict)

pred_model = NeuralNetwork().to(device)
print(pred_model)     
x1 = torch.rand(1, 28, 28, device=device)
logits = pred_model(x1)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
    


# print(max(pred[0]))
#End of program
#***************************************************************************************************************