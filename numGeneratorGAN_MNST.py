import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale the range of the images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# Define the dimensions of the images
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
channels = X_train.shape[3]

# Define the shape of the noise vector
z_dim = 100

# Generator model
def build_generator(z_dim):
    model = Sequential()
    
    # Hidden layer
    model.add(Dense(128 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    # Upsampling
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    # Output layer
    model.add(Conv2DTranspose(channels, kernel_size=5, activation='tanh', padding='same'))
    return model

# Discriminator model
def build_discriminator(img_shape):
    model = Sequential()
    
    # Hidden layer
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.01))
    
    # Hidden layer
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    # Flatten
    model.add(Flatten())
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator((img_rows, img_cols, channels))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = build_generator(z_dim)

# Generat image to be used as input
z = Input(shape=(z_dim,))
img = generator(z)

# Keep the discriminators parameters constant
discriminator.trainable = False
validity = discriminator(img)

# Combine the models
gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Training loop
def train_gan(X_train, epochs, batch_size, save_interval):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Loss and accuracy arrays
    d_loss_history = []
    g_loss_history = []
    
    for epoch in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(noise)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        g_loss = gan.train_on_batch(noise, valid)
        
        d_loss_history.append(d_loss[0])
        g_loss_history.append(g_loss)
        
        # Print progress
        print(f"Epoch {epoch}, [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%], [G loss: {g_loss}]")
        
        # Save generated images at desird interval
        if epoch % save_interval == 0:
            save_generated_images(epoch, generator)
    
    # Plot loss history
    plot_loss_history(d_loss_history, g_loss_history)

# Function to save generated images
def save_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, z_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
    plt.close()

# Function to plot the loss history
def plot_loss_history(d_loss_history, g_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_history, label='Discriminator loss')
    plt.plot(g_loss_history, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss')
    plt.legend()
    plt.savefig('gan_loss_history.png')
    plt.show()

# Hyperparameters
epochs = 300000
batch_size = 128
save_interval = 1000

# Train the GAN
train_gan(X_train, epochs, batch_size, save_interval)
