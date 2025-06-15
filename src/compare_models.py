import numpy as np
import time
import tensorflow as tf

# Load validation data
X_val = np.load('data/signals.npy') 
Y_val = np.load('data/labels.npy') 

# Load Keras model
keras_model = tf.keras.models.load_model('model/signal_classifier.keras')
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model/signal_classifier.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def evaluate_keras_model(model, X, Y):
    correct = 0
    total_time = 0.0
    num_samples = len(X)

    for i in range(num_samples):
        x = np.expand_dims(X[i], axis=0)  # shape (1, seq_len, 1)
        start = time.time()
        preds = model.predict(x, verbose=0)
        end = time.time()
        pred_label = np.argmax(preds)
        if pred_label == Y[i]:
            correct += 1
        total_time += (end - start)

    accuracy = correct / num_samples
    avg_time = total_time / num_samples
    return accuracy, avg_time

def evaluate_tflite_model(interpreter, X, Y):
    correct = 0
    total_time = 0.0
    num_samples = len(X)

    for i in range(num_samples):
        input_data = np.expand_dims(np.expand_dims(X[i], axis=-1), axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start = time.time()
        interpreter.invoke()
        end = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_label = np.argmax(output_data)

        if pred_label == Y[i]:
            correct += 1
        total_time += (end - start)

    accuracy = correct / num_samples
    avg_time = total_time / num_samples
    return accuracy, avg_time

# Evaluate Keras model
print("Evaluating Keras model...")
keras_acc, keras_avg_time = evaluate_keras_model(keras_model, X_val, Y_val)
print(f"Keras model accuracy: {keras_acc*100:.2f}%")
print(f"Keras model average inference time per sample: {keras_avg_time*1000:.3f} ms\n")

# Evaluate TFLite model
print("Evaluating TFLite model...")
tflite_acc, tflite_avg_time = evaluate_tflite_model(interpreter, X_val, Y_val)
print(f"TFLite model accuracy: {tflite_acc*100:.2f}%")
print(f"TFLite model average inference time per sample: {tflite_avg_time*1000:.3f} ms\n")

# Summary
print("Summary Comparison:")
print(f"Keras accuracy: {keras_acc*100:.2f}%, avg latency: {keras_avg_time*1000:.3f} ms")
print(f"TFLite accuracy: {tflite_acc*100:.2f}%, avg latency: {tflite_avg_time*1000:.3f} ms")