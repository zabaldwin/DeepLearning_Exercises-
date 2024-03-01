import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(trainloader, criterion, optimizer, num_epochs=5):
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print('[Epoch %d] Loss: %.3f | Accuracy: %.2f%%' % (epoch + 1, epoch_loss, epoch_accuracy))

    return losses, accuracies

# Function to evaluate the model
def evaluate_model(testloader):
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on the test set: %.2f%%' % accuracy)
    
    return predictions

# Define hyperparameters for tuning
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
num_epochs = 5

best_accuracy = 0.0
best_lr = None
best_batch_size = None
best_losses = None
best_accuracies = None
best_predictions = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Define the model
        net = CNN()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # Train the model
        print(f'Training model with LR={lr}, Batch Size={batch_size}')
        losses, accuracies = train_model(trainloader, criterion, optimizer, num_epochs)
        
        # Evaluate the model
        predictions = evaluate_model(testloader)
        
        # Check if current model is the best so far
        if accuracies[-1] > best_accuracy:
            best_accuracy = accuracies[-1]
            best_lr = lr
            best_batch_size = batch_size
            best_losses = losses
            best_accuracies = accuracies
            best_predictions = predictions

print("Best Accuracy:", best_accuracy)
print("Best Learning Rate:", best_lr)
print("Best Batch Size:", best_batch_size)

# Plot training metrics for the best model
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(best_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Plot confusion matrix for the best model
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
cm = confusion_matrix(testset.targets, best_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.title('Confusion Matrix')
plt.show()
