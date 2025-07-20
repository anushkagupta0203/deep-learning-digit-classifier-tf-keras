# 🧠 Handwritten Digit Classification using Deep Learning (MNIST + TensorFlow/Keras)

## 📌 Overview
This project builds a deep learning model using TensorFlow and Keras to classify handwritten digits from the **MNIST** dataset. The goal is to train a neural network that accurately identifies digits (0–9) from 28x28 grayscale images.

---

## 🗃️ About the MNIST Dataset
The **MNIST (Modified National Institute of Standards and Technology)** dataset is a classic benchmark for image classification.

- **Images**: 70,000 grayscale images
  - 60,000 for training
  - 10,000 for testing
- **Image Size**: 28x28 pixels (flattened to 784 features)
- **Labels**: Digits from 0 to 9
- **Format**: Each image is labeled and cleaned, making it ideal for deep learning beginners
- **Final Accuracy obtained**: 98.3299%
---

## 🧰 Tools & Technologies Used

| Tool          | Description |
|---------------|-------------|
| **TensorFlow** | Open-source deep learning framework developed by Google |
| **Keras**      | High-level neural networks API, integrated with TensorFlow |
| **Python**     | Programming language for implementing the model |
| **NumPy/Matplotlib** | For data handling and visualization |

---

## 🏗️ Model Architecture

```python
model = keras.Sequential([
    keras.layers.Dense(1000, input_shape=(784,), activation='relu'),
    keras.layers.Dense(100, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')  # Output layer
])
