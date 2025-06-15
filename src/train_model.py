# train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

# Load data
X = np.load("data/signals.npy")
Y = np.load("data/labels.npy")

# Normalize data
X_Norm = X / np.max(np.abs(X), axis=1, keepdims=True)
X_Norm = X[..., np.newaxis]  # shape becomes (samples, 128, 1)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X_Norm, Y, test_size=0.2, random_state=42)

# Model definition
model = models.Sequential([
    layers.Conv1D(16, 3, activation='relu', input_shape=(128, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(Y)), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Save
os.makedirs("model", exist_ok=True)
model.save("model/signal_classifier.keras")