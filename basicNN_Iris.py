import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Iris dataset
iris = load_iris()
X, Y = iris.data, iris.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(categories='auto')
Y_onehot = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

# Sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(s):
    return s * (1 - s)

# Calculate accuracy
def calculate_accuracy(X, Y, W1, b1, W2, b2):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    Y_pred = sigmoid(output_layer_input)
    Y_pred_class = np.argmax(Y_pred, axis=1)
    Y_true_class = np.argmax(Y, axis=1)
    accuracy = np.mean(Y_pred_class == Y_true_class)
    return accuracy

# Initialize weights and biases
np.random.seed(42)
weights1 = np.random.rand(X_train.shape[1], 5)
weights2 = np.random.rand(5, Y_train.shape[1])
bias1 = np.zeros(5)
bias2 = np.zeros(Y_train.shape[1])

# Training hyperparameters
epochs = 200
learning_rate = 0.1


losses = []
accuracies = []

# Training
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train, weights1) + bias1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, weights2) + bias2
    a2 = sigmoid(z2)

    # Backward pass
    a2_error = a2 - Y_train
    a2_delta = a2_error * sigmoid_derivative(a2)
    a1_error = a2_delta.dot(weights2.T)
    a1_delta = a1_error * sigmoid_derivative(a1)

    # Update weights and biases
    weights2 -= learning_rate * a1.T.dot(a2_delta)
    bias2 -= learning_rate * np.sum(a2_delta, axis=0)
    weights1 -= learning_rate * X_train.T.dot(a1_delta)
    bias1 -= learning_rate * np.sum(a1_delta, axis=0)
    
     # Compute loss and accuracy
    loss = np.mean(np.square(Y_train - a2))
    losses.append(loss)

    predictions = np.argmax(a2, axis=1)
    labels = np.argmax(Y_train, axis=1)
    accuracy = np.mean(predictions == labels)
    accuracies.append(accuracy)

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        loss = np.mean(np.square(Y_train - a2))
        print(f'Epoch {epoch}, loss: {loss:.4f}')

# Evaluate training set accuracy
predicted = sigmoid(np.dot(sigmoid(np.dot(X_train, weights1) + bias1), weights2) + bias2)
predictions = np.argmax(predicted, axis=1)
labels = np.argmax(Y_train, axis=1)
accuracy = np.mean(predictions == labels)
print(f'Training accuracy: {accuracy:.2f}')

# Plotting
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Import more variables for cofusion matrix visual
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predictions on the test set
z1_test = np.dot(X_test, weights1) + bias1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, weights2) + bias2
a2_test = sigmoid(z2_test)
predictions_test = np.argmax(a2_test, axis=1)
labels_test = np.argmax(Y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(labels_test, predictions_test)

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

