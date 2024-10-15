import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate a simple dataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Create a dataset
X, y = make_moons(n_samples=1000, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create and compile a model with the specified activation function
def create_model(activation_function):
    model = keras.Sequential([
        layers.Dense(10, activation=activation_function, input_shape=(2,)),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# List of activation functions to test
activation_functions = ['sigmoid', 'tanh', 'relu']

# Store results
results = {}

# Train models with different activation functions
for activation in activation_functions:
    print(f"\nTraining model with {activation} activation function...")
    model = create_model(activation)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
   
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[activation] = (history.history, test_loss, test_acc)

# Plot the results
plt.figure(figsize=(12, 6))
for activation, (history, test_loss, test_acc) in results.items():
    plt.plot(history['accuracy'], label=f'{activation} Train Accuracy')
    plt.plot(history['val_accuracy'], label=f'{activation} Val Accuracy')

plt.title('Model Accuracy per Activation Function')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Using Softmax for multi-class classification (for demonstration)
def create_softmax_model():
    model = keras.Sequential([
        layers.Dense(10, activation='softmax', input_shape=(2,)),
        layers.Dense(3, activation='softmax')  # Output layer for multi-class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Convert to multi-class for softmax example
X_multi, y_multi = make_moons(n_samples=1000, noise=0.1)
y_multi = (y_multi * 2).astype(int)  # Convert binary to two classes for demo
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Train the model with softmax activation
print("\nTraining model with Softmax activation function...")
softmax_model = create_softmax_model()
softmax_history = softmax_model.fit(X_train_multi, y_train_multi, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate softmax model
softmax_test_loss, softmax_test_acc = softmax_model.evaluate(X_test_multi, y_test_multi, verbose=0)

# Print softmax model results
print(f'Softmax Model - Test Loss: {softmax_test_loss}, Test Accuracy: {softmax_test_acc}')

