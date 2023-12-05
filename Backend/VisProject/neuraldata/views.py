from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework import viewsets
from django.core.files.base import ContentFile
import pickle
from .models import NeuralNetwork
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchsummary import summary

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(7*7, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 10)
        self.register_hooks = True  # Flag to control hook registration

    def forward(self, x):
        x = x.view(-1, 7*7)
        x = self.fc1(x)
        if self.register_hooks:
            x.register_hook(get_gradient('fc1_grad'))
        x = self.relu1(x)
        if self.register_hooks:
            x.register_hook(get_gradient('relu1_grad'))
        x = self.fc2(x)
        if self.register_hooks:
            x.register_hook(get_gradient('fc2_grad'))
        x = self.relu2(x)
        if self.register_hooks:
            x.register_hook(get_gradient('relu2_grad'))
        x = self.fc3(x)
        if self.register_hooks:
            x.register_hook(get_gradient('fc3_grad'))
        return x

def get_layer_order(model):
    layer_order = ['input']
    for name, _ in model.named_children():
        layer_order.append(name)
    return layer_order

# Function to register forward hook
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


# Function to register backward hook
def get_gradient(name):
    def hook(grad):
        nn_model = NeuralNetwork.objects.get(name="model"+str(NeuralNetwork.objects.count()))
        gradients = json.loads(nn_model.gradients)
        gradients[name] = grad.detach()
    return hook

# Function to get network shape
def get_network_shape(model):
    shape = {}
    first_layer = True

    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            # Record input size for the first linear layer
            if first_layer:
                shape['input_size'] = layer.in_features
                first_layer = False

            # Record the number of output features (neurons)
            shape[name] = layer.out_features

        elif isinstance(layer, nn.ReLU):
            # Set the size of ReLU layer equal to the size of the last linear layer
            # This assumes that a ReLU layer follows a linear layer
            prev_layer = list(model.named_children())[list(model.named_children()).index((name, layer)) - 1][1]
            if isinstance(prev_layer, nn.Linear):
                shape[name] = prev_layer.out_features
            else:
                shape[name] = 'unknown'
        else:
            shape[name] = 'unknown'  # For other types of layers

    return shape



def init():
    # Instantiate the model
    model = SimpleNet()

    # Registering forward hooks
    activations = {}
    for name, layer in model.named_children():
        layer.register_forward_hook(get_activation(name))

    # Storage for gradients
    gradients = {}

    # Training loop
    model.register_hooks = True  # Enable hook registration
    
    return model, activations, gradients

def train(model, activations, gradients, criterion, optimizer, train_loader, test_loader):
    for i, (images, labels) in enumerate(train_loader):
        activations = {}  # Storage for activations

        activations['input'] = images.view(32, -1).detach()  # Store input images
        # Forward pass
        images.requires_grad = True  # Set requires_grad to True for input images
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Storing gradients of input data
        gradients['input_grad'] = images.grad.view(32, -1).detach()

        # Storing gradients of parameters
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradients[name] = parameter.grad

        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch, Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            gradient_data = {}
            gradients_data_shape = {}

            for key, value in gradients.items():
                # Average only the activation gradients (identified by 'grad' in their names)
                if 'grad' in key:
                    gradient_data[key] = value.mean(dim=0).numpy().tolist()
                    gradients_data_shape[key] = list(value.mean(dim=0).size())
                else:
                    gradient_data[key] = value.numpy().tolist()
                    gradients_data_shape[key] = list(value.size())
                    
            
            # Test the model
            model.eval()  # Set the model to evaluation mode
            model.register_hooks = False  # Disable hook registration
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_loss = criterion(outputs, labels).item()
                accurracy = correct / total
                print(f'Accuracy of the model on the 10000 test images: {100 * accurracy} %')
                    
                    
            averaged_activation_data = {key: value.mean(dim=0).numpy().tolist() for key, value in activations.items()}
            averaged_activation_data_shape = {key: list(value.mean(dim=0).size()) for key, value in activations.items()}

            layer_order = get_layer_order(model)

            data_to_send = {
                "activations": averaged_activation_data,
                "activations_shape": averaged_activation_data_shape,
                "gradients": gradient_data,
                "gradients_shape": gradients_data_shape,
                "loss": loss.item(),  # Include the loss
                "layer_order": layer_order,
                "test_loss": test_loss,
            }

            return data_to_send

@api_view(['GET'])
def init_model(request):
    model, activations, gradients = init()
    
    # Save the model, activations, and gradients to the database
    ## name the model as "model"+str(NeuralNetwork.objects.count()+1)
    nn_model = NeuralNetwork(name="model"+str(NeuralNetwork.objects.count()+1))
    nn_model.state_dict = pickle.dumps(model.state_dict())
    nn_model.activations = json.dumps(activations)
    nn_model.activations_shape = json.dumps(get_network_shape(model))
    nn_model.gradients = json.dumps(gradients)
    nn_model.gradients_shape = json.dumps(get_network_shape(model))
    nn_model.save()
    
    return Response({"message": "Model initialized."})

@api_view(['GET'])
def train_model(request):
    # Retrieve the model state_dict from the database
    nn_model = NeuralNetwork.objects.get(name="model"+str(NeuralNetwork.objects.count()))
    state_dict = pickle.loads(nn_model.state_dict)
    # Instantiate the model and load the state_dict
    model = SimpleNet()
    model.load_state_dict(state_dict)
    activations = json.loads(nn_model.activations)
    gradients = json.loads(nn_model.gradients)
    # Registering forward hooks
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((7, 7)),  # Resize the images to 10x10
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # Train the model for one epoch
    data = train(model, activations, gradients, criterion, optimizer, train_loader, test_loader)
    return Response(data)
