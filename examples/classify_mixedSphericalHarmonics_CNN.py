import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Generate mixed spherical harmonics
def generate_mixed_spherical_harmonics(num_samples=100, grid_size=64):
    l_values = [0, 1, 2, 3]
    m_values_range = [-2, -1, 0, 1, 2]

    X = np.zeros((num_samples, grid_size, grid_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, len(l_values)), dtype=np.float32)

    for i in range(num_samples):
        l = np.random.choice(l_values)
        m = np.random.choice([m_val for m_val in m_values_range if abs(m_val) <= l])

        theta = np.linspace(0, np.pi, num=grid_size)
        phi = np.linspace(0, 2*np.pi, num=grid_size)
        phi, theta = np.meshgrid(phi, theta)

        Ylm = sph_harm(m, l, phi, theta)
        X[i, :, :, 0] = np.abs(Ylm.real)

        label_index = l_values.index(l)
        y[i, label_index] = 1

    return X, y

# Plot spherical harmonics
def plot_spherical_harmonics(l, m, grid_size=100):
    theta = np.linspace(0, np.pi, num=grid_size)
    phi = np.linspace(0, 2*np.pi, num=grid_size)
    phi, theta = np.meshgrid(phi, theta)

    Ylm = sph_harm(m, l, phi, theta)

    x = np.abs(Ylm) * np.sin(theta) * np.cos(phi)
    y = np.abs(Ylm) * np.sin(theta) * np.sin(phi)
    z = np.abs(Ylm) * np.cos(theta)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.viridis)
    plt.title(f'Spherical Harmonics l={l}, m={m}')
    plt.colorbar(surf)
    plt.show()

# Example plotting for l=2, m=1
plot_spherical_harmonics(l=2, m=1)

# Convert one-hot encoded labels to class indices
def convert_to_class_indices(y):
    return np.argmax(y, axis=1)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, criterion, and optimizer
model = SimpleCNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define functions for computing and visualizing gradients
def compute_visualization_gradients(model, inputs):
    model.eval()
    inputs.requires_grad = True
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    score = outputs[torch.arange(outputs.shape[0]), predictions].sum()
    score.backward()
    gradients = inputs.grad.data
    return gradients

def compute_gradients(model, inputs):
    inputs = inputs.permute(0, 3, 1, 2)
    inputs.requires_grad = True
    model.eval()
    outputs = model(inputs)
    target_class_index = 1
    score = outputs[:, target_class_index].sum()
    score.backward()
    gradients = inputs.grad
    return gradients

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

# Generate data and split into train/test sets
X, y = generate_mixed_spherical_harmonics(num_samples=1000, grid_size=32)
y_class_indices = convert_to_class_indices(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_class_indices, test_size=0.2, random_state=42)

# Create DataLoader objects for train/test sets
train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Training loop
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs = inputs.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate model on test set
evaluate_model(model, test_loader)

# Visualize gradients for a sample input
sample_input, _ = next(iter(test_loader))
sample_input = sample_input[:1]
gradients = compute_gradients(model, sample_input)

# Visualize the gradient heatmap
plt.imshow(gradients.abs().squeeze().numpy(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Gradient Heatmap')
plt.show()

# Plot spherical harmonics and corresponding gradients
grid_size = 32
lm_pairs = [(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
num_plots = len(lm_pairs)
cols = 3
rows = num_plots
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for i, (l, m) in enumerate(lm_pairs):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, grid_size), np.linspace(0, 2 * np.pi, grid_size))
    Ylm = sph_harm(m, l, phi, theta)
    spherical_harmonic_magnitude = np.abs(Ylm.real)
    min_val = spherical_harmonic_magnitude.min()
    max_val = spherical_harmonic_magnitude.max()
    if max_val - min_val > 0:
        sample_harmonic_normalized = (spherical_harmonic_magnitude - min_val) / (max_val - min_val)
    else:
        sample_harmonic_normalized = spherical_harmonic_magnitude
    sample_input = torch.tensor(sample_harmonic_normalized).float().unsqueeze(0).unsqueeze(0)
    sample_input = sample_input.permute(0, 2, 3, 1)
    sample_input = sample_input.permute(0, 3, 1, 2)
    gradient_heatmap = compute_visualization_gradients(model, sample_input).abs().squeeze().numpy()
    ax = axes[i, 0]
    ax.imshow(spherical_harmonic_magnitude, cmap='viridis')
    ax.set_title(f'Spherical Harmonic l={l}, m={m}')
    ax.axis('off')
    ax = axes[i, 1]
    ax.imshow(sample_harmonic_normalized, cmap='gray')
    ax.set_title(f'Input l={l}, m={m}')
    ax.axis('off')
    ax = axes[i, 2]
    ax.imshow(gradient_heatmap, cmap='hot')
    ax.set_title('Gradient Heatmap')
    ax.axis('off')

plt.show()
