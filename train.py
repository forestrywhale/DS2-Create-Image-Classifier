# Imports here
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
from workspace_utils import active_session

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('datadir', help='Data directory path.')
parser.add_argument('--save_dir', help='Directory path to save checkpoint .pth file.', default = 'checkpoint.pth')
parser.add_argument('--arch', help='Choosing models. vgg16 & vgg13 are available')
parser.add_argument('--epochs', help='Number of epochs.', default=1)
parser.add_argument('--learning_rate', help='Set learning rate.', default=0.001)
parser.add_argument('--hidden1_units', help='Set number of nodes in hidden layer 1.', default=1024)
parser.add_argument('--hidden2_units', help='Set number of nodes in hidden layer 2.', default=256)
parser.add_argument('--gpu', help='Binary value. If you want to use GPU', default = True)
args = parser.parse_args()

data_dir = args.datadir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print(train_dir)

num_classes = 102
np.random.seed(123)

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    # note, it might be important to change it to tensor just right before the normalization.
    # image transform/changes -> to tensor -> numerical transform
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),

    "valid_test": transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# TODO: Load the datasets with ImageFolder
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_data = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms["train"])
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms["valid_test"])
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.ImageFolder(test_dir, transform=data_transforms["valid_test"])
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if {torch.cuda.is_available() and args.gpu} else "cpu")

if args.arch == "vgg13":
    vgg = models.vgg13(pretrained=True)
else:
    vgg = models.vgg16(pretrained=True)

# hyperparameters
imputsize = 25088
h1size = args.hidden1_units
h2size = args.hidden2_units
outsize = num_classes
alpha = args.learning_rate

# Freeze parameters so we don't backprop through them
for param in vgg.parameters():
    param.requires_grad = False

# define a new classifier
# 2 hidden layer
# 1 output layer
from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('h1', nn.Linear(imputsize, h1size)),
    ('relu1', nn.ReLU()),
    ('drop1', nn.Dropout(p=0.5)),
    ('h2', nn.Linear(h1size, h2size)),
    ('relu2', nn.ReLU()),
    ('o3', nn.Linear(h2size, outsize)),
    ('output', nn.LogSoftmax(dim=1))]))

vgg.classifier = classifier
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(vgg.classifier.parameters(), lr=alpha)
vgg.to(device);

with active_session():
    # TODO: Build and train your network
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 20

    i = 0
    traininglosslist = []
    validlosslist = []
    validacculist = []
    for epoch in range(epochs):
        for inputs, labels in iter(train_data_loader):
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = vgg.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                vgg.eval()
                with torch.no_grad():
                    for inputs, labels in iter(valid_data_loader):
                        # print("hello")
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = vgg.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                traininglosslist.append(running_loss / print_every)
                validlosslist.append(valid_loss / len(valid_data_loader))
                validacculist.append(accuracy / len(valid_data_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"valid loss: {valid_loss/len(valid_data_loader):.3f}.. "
                      f"valid accuracy: {accuracy/len(valid_data_loader):.3f}")
                running_loss = 0
                vgg.train()

hidden_layers = [h1size,h2size]
batch_size = imputsize
lr = alpha
model_name = args.arch
epoch = args.epochs
vgg.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': (3, 224, 224),
              'output_size': 102,
              'hidden_layers': hidden_layers,
              'batch_size': batch_size,
              'learning_rate': lr,
              'model_name': model_name,
              'state_dict': vgg.state_dict(),
              'optimizer': optimizer.state_dict(),
              'classifier':vgg.classifier,
              'epoch': epochs,
              'class_to_idx': vgg.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

torch.save(checkpoint, args.save_dir)