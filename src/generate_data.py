import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
num_samples = 1000     # per class
sample_length = 128    # number of time steps per sample
classes = ['sine', 'square', 'noise']
output_dir = 'data'

os.makedirs(output_dir, exist_ok=True)

def generate_sine_wave():
    f = np.random.uniform(1, 10)
    t = np.linspace(0, 1, sample_length)
    return np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))

def generate_square_wave():
    f = np.random.uniform(1, 10)
    t = np.linspace(0, 1, sample_length)
    return np.sign(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi)))

def generate_noise():
    return np.random.normal(0, 1, sample_length)

signal_generators = {
    'sine': generate_sine_wave,
    'square': generate_square_wave,
    'noise': generate_noise
}

# Generate data
X = []
Y = []

for idx, label in enumerate(classes):
    print(f"Generating {label} signals...")
    for _ in range(num_samples):
        signal = signal_generators[label]()
        X.append(signal)
        Y.append(idx)

X = np.array(X)
Y= np.array(Y)

np.save(os.path.join(output_dir, 'signals.npy'), X)
np.save(os.path.join(output_dir, 'labels.npy'), Y)

print("Signal data generated and saved.")

# Load and Plot

X_loaded = np.load("data/signals.npy")
Y_loaded = np.load("data/labels.npy")

for idx, label in enumerate(classes):
    example_signal = X_loaded[Y_loaded == idx][0]
    plt.plot(example_signal, label=label)
plt.legend()
plt.title("Generated Signals")
plt.show()