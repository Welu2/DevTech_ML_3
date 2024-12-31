import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Reshape labels
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Class names
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_data(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()

plot_data(X_train, y_train, 0)
plot_data(X_train, y_train, 7)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# ANN model
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = ann.fit(X_train, y_train, epochs=5)

print("ANN Training History:", history.history)
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# CNN model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn_history = cnn.fit(X_train, y_train, epochs=10)
print("CNN Training History:", cnn_history.history)

cnn_eval = cnn.evaluate(X_test, y_test)
print(f"CNN Test Loss and Accuracy: {cnn_eval}")
y_pred_cnn = cnn.predict(X_test)
y_pred_classes_cnn = [np.argmax(element) for element in y_pred_cnn]

# Print classification report for CNN model
print("Classification Report (CNN): \n", classification_report(y_test, y_pred_classes_cnn))

# Example of comparing predicted vs actual values
index = 2
plot_data(X_test, y_test, index)
print("Predicted Class: ", classes[y_pred_classes_cnn[index]])
print("Actual Class: ", classes[y_test[index]])
