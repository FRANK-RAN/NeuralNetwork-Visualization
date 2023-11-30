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



# Instantiate the model
model = SimpleNet()

# summary(model, input_size=(1, 28, 28))  # Adjust input_size based on your model's input


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
        # Forward pass
        images.requires_grad = True  # Set requires_grad to True for input images
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Storing gradients of parameters
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradients[name] = parameter.grad

        optimizer.step()



        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # Example of sending data (add this inside your training loop)
            # activation_data = {key: value.numpy().tolist() for key, value in activations.items()}
            # gradient_data = {key: value.numpy().tolist() for key, value in gradients.items()}

            # requests.post("http://localhost:8000/activations/", json={"data": activation_data})
            # requests.post("http://localhost:8000/gradients/", json={"data": gradient_data})
            
            # Average activations and gradients over the batch and convert to lists
            averaged_activation_data = {key: value.mean(dim=0).numpy().tolist() for key, value in activations.items()}
            averaged_gradient_data = {key: value.mean(dim=0).numpy().tolist() for key, value in gradients.items()}

            # Post the averaged data
            requests.post("http://localhost:8000/activations/", json={"data": averaged_activation_data})
            requests.post("http://localhost:8000/gradients/", json={"data": averaged_gradient_data})
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




