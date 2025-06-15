# TinyML Signal Classifier

This project demonstrates a simple pipeline for generating synthetic signals, training a lightweight neural network to classify them, converting the model to TensorFlow Lite format for edge deployment, and running inference using the TFLite interpreter.

## Motivation

In embedded applications such as telecommunications, IoT, and edge computing, lightweight AI models are essential due to limited hardware resources. This project demonstrates the use of TensorFlow Lite Interpreter to deploy compact, efficient signal classification models suitable for such environments.

## Project Structure

```
├── src/  
│   ├── generate_data.py         # Create synthetic signals  
│   ├── train_model.py           # Train a small model (Keras)  
│   ├── convert_tflite.py        # Convert to TFLite + quantize  
│   ├── run_inference.py         # Simulate edge inference on PC  
│   ├── compare_models.py        # Compare Karas and TFLite model 
│  
├── notebooks/  
│   └── visualize_signals.ipynb  # Plotting and understanding data  
│  
├── README.md                    # Project description and instructions  
├── requirements.txt             # Python dependencies  
└── model/                       # Stores trained and converted models  
```

## Features

- Synthetic data generation for multiple signal types
- Compact CNN architecture tailored for small datasets and embedded inference
- TensorFlow Lite conversion with quantization for efficient edge execution
- Example of signal classification inference using TFLite Interpreter in Python

## Getting Started

### Clone the repository:
```bash
git clone https://github.com/bhavanithya21/tinyml-signal-classifier.git
cd tinyml-signal-classifier
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Generate data and train the model:
```bash
python src/generate_data.py
python src/train_model.py
```
> Note: Running these scripts will create the data/ and model/ folders automatically in the project directory.

### Convert model to TensorFlow Lite format:
```bash
python src/convert_tflite.py
```
### Run inference with the TFLite model:
```bash
python src/run_inference.py
```
### Run Comparison for Karas and TFLite Model:
```bash
python src/compare_models.py
```
## Requirements

This project uses Python 3.11.9 and requires the following libraries:

- `numpy` – for numerical operations and signal generation
- `matplotlib` – for visualizing synthetic signals
- `scikit-learn` – for dataset splitting
- `tensorflow` – for building, training, and converting the model using TensorFlow Lite
> TensorFlow Lite functionality (conversion and inference) is included within the main `tensorflow` package.

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Model Performance Comparison

This section compares the TensorFlow Keras model and the TensorFlow Lite (TFLite) model in terms of accuracy and inference latency on the validation dataset.

| Model           | Accuracy (%) | Average Inference Time per Sample (ms) |
|-----------------|--------------|---------------------------------------|
| Keras (TensorFlow) | 100       | 30.754                                  |
| TensorFlow Lite  | 100        | 0.001                                   |

> **Note:** The TFLite model is optimized for edge deployment, offering significantly faster inference with minimal accuracy loss, making it suitable for resource-constrained environments such as embedded IoT and telecom devices.

## Future Work

- Extend to additional signal types and real RF datasets
- Implement FPGA/embedded hardware deployment examples
- Explore advanced AI models and pruning for better edge performance

## License

This project is licensed under the MIT License.
