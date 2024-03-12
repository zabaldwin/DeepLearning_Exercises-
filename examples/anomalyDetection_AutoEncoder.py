import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import uproot

# Load Data
def load_data(file_path):
    # Open the ROOT file and get the tree
    with uproot.open(file_path) as root_file:
        tree = root_file['tree']
        # Convert the data from the tree into a NumPy array
        arrays = tree.arrays(['response'])
        # Extract the 'response' array from the dictionary of arrays
        data = arrays['response']
    # Convert to NumPy array
    return np.array(data)

# Preprocess Data
def preprocess_data(data):
    # Normalize the data to have zero mean and unit variance
    data_mean = np.mean(data)
    data_std = np.std(data)
    normalized_data = (data - data_mean) / data_std

    # Reshape the data 
    normalized_data = normalized_data[:, np.newaxis]

    train_data, val_data = train_test_split(normalized_data, test_size=0.2, random_state=42)

    return np.array(train_data), np.array(val_data)

# Build Autoencoder Model
def build_autoencoder(input_shape):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu')
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(input_shape, activation='linear')
    ])

    autoencoder = tf.keras.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

# Train Autoencoder
def train_autoencoder(autoencoder, train_data, val_data, epochs=10, batch_size=32):
    autoencoder.fit(train_data, train_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(val_data, val_data))

# Evaluation
def evaluate_autoencoder(autoencoder, val_data):
    # Reshape validation data to match the expected input shape of the model
    val_data_reshaped = val_data.reshape(-1, 1)  # Reshape to (None, 1)

    # Predict reconstructions using the autoencoder
    reconstructions = autoencoder.predict(val_data_reshaped)

    # Calculate reconstruction errors
    reconstruction_errors = np.mean(np.square(val_data_reshaped - reconstructions), axis=1)

    return reconstruction_errors, reconstructions

# Load data
file_path = 'normalData.root'
data = load_data(file_path)

train_data, val_data = preprocess_data(data)

input_shape = train_data.shape[1]
autoencoder = build_autoencoder(input_shape)

train_autoencoder(autoencoder, train_data, val_data)

reconstruction_errors, reconstructions = evaluate_autoencoder(autoencoder, val_data)

# Assess the model's ability to detect anomalies
threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
anomalies_detected = sum(reconstruction_errors > threshold)
print(f"Anomalies detected in validation data using thresholding: {anomalies_detected}")

# Calculate area under the ROC Curve
auc_score = roc_auc_score(np.where(reconstruction_errors > threshold, 1, 0), reconstruction_errors)
print(f"Area under the ROC Curve: {auc_score}")

# Calculate Precision, Recall, and F1 Score
precision, recall, f1_score, _ = precision_recall_fscore_support(
    np.where(reconstruction_errors > threshold, 1, 0), 
    np.where(reconstruction_errors > threshold, 1, 0), 
    average='binary'
)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

# Visual inspection by plotting original and reconstructed data points
plt.figure(figsize=(10, 5))
plt.plot(val_data, label='Original Data')
plt.plot(reconstructions, label='Reconstructed Data')
plt.title('Original vs Reconstructed Data')
plt.xlabel('Index')
plt.ylabel('Response')
plt.legend()
plt.show()

# Testing (Optional)
test_file_path = 'mixedData.root'  # Path to the ROOT file containing the test data
test_data = load_data(test_file_path)
normalized_test_data = (test_data - np.mean(train_data)) / np.std(train_data)  # Normalize

test_reconstruction_errors, test_reconstructions = evaluate_autoencoder(autoencoder, normalized_test_data)

# Adjust threshold based on validation data
test_threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)

# Anomalies detected in testing data using adjusted threshold
test_anomalies_detected = sum(test_reconstruction_errors > test_threshold)
print(f"Anomalies detected in testing data using adjusted threshold: {test_anomalies_detected}")

plt.figure(figsize=(10, 5))
plt.plot(normalized_test_data, label='Original Data')
plt.plot(test_reconstructions, label='Reconstructed Data')
plt.title('Original vs Reconstructed Data on Test Data')
plt.xlabel('Index')
plt.ylabel('Response')
plt.legend()
plt.show()
