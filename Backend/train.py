import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchsummary import summary
import requests

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 10)
        self.register_hooks = True  # Flag to control hook registration

    def forward(self, x):
        x = x.view(-1, 28*28)
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

# Function to register forward hook
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


# Function to register backward hook
def get_gradient(name):
    def hook(grad):
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



# Instantiate the model
model = SimpleNet()



# Registering forward hooks
activations = {}
for name, layer in model.named_children():
    layer.register_forward_hook(get_activation(name))

# Storage for gradients
gradients = {}

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Training loop
num_epochs = 3
model.register_hooks = True  # Enable hook registration


for epoch in range(num_epochs):
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

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
                    
                    
            averaged_activation_data = {key: value.mean(dim=0).numpy().tolist() for key, value in activations.items()}
            averaged_activation_data_shape = {key: list(value.mean(dim=0).size()) for key, value in activations.items()}

            data_to_send = {
                "activations": averaged_activation_data,
                "activations_shape": averaged_activation_data_shape,
                "gradients": gradient_data,
                "gradients_shape": gradients_data_shape,
                "loss": loss.item()  # Include the loss
            }

            requests.post("http://localhost:8000/trainingdata/", json=data_to_send)



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

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

print('Finished Training and Testing')




