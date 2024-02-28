import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, ELU
from tensorflow.keras.utils import to_categorical

# Preprocess data
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalizes to [0, 1] by scaling the pixel values
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

# Define a function to create a model
def create_model(activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128),
    ])
    
    # If the activation is a string, use it directly in the dense layer
    if isinstance(activation, str):
        model.add(Dense(128, activation=activation))
    # Otherwise, assume it's a layer instance and add it separately
    else:
        model.add(Dense(128))
        model.add(activation())
    
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Activation functions to experiment with
activation_functions = {'ReLU': 'relu', 'Sigmoid': 'sigmoid', 'Tanh': 'tanh', 'LeakyReLU': LeakyReLU, 'ELU': ELU}

# Train models and collect histories
histories = {}
for name, activation in activation_functions.items():
    print(f"Training model with {name} activation function...")
    model = create_model(activation=activation)
    history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, verbose=0)
    histories[name] = history

# Determine the bst activation function based on validation accuracy
best_activation = None
highest_val_accuracy = 0

for name, history in histories.items():
    val_accuracy = max(history.history['val_accuracy'])  # Get the highest validation accuracy for this activation
    if val_accuracy > highest_val_accuracy:
        highest_val_accuracy = val_accuracy
        best_activation = name

print(f"\nThe best activation function is {best_activation} with a validation accuracy of {highest_val_accuracy:.4f}.")

# Plotting
plt.figure(figsize=(14, 8))

for name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{name} Training')
    plt.plot(history.history['val_accuracy'], '--', label=f'{name} Validation')

plt.title('Training and Validation Accuracy by Activation Function')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
