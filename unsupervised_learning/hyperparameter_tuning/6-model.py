""" Defines a simple model to be trained on IRIS dataset and optimized
    with Bayessian Optimization

Model Architecture:
    1. Dense Layer, input shape (28, 28)

Requires:
    - tensorflow
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def build_simple_model(learning_rate, layers, dropout_rate, batch_size, l2_reg, ishape):
    """ Builds a simple model
        - learning_rate: learning rate for the model
        - layers: number of nodes for each layer
        - dropout_rate: dropout rate
        - batch_size: number of samples to be processed during training
        - l2_reg: L2 regularization parameter
        - ishape: input shape
    """
    model = Sequential([])

    # Add the first layer with L2 regularization
    L0 = Dense(units=int(layers[0]), activation="relu", input_shape=(ishape,), kernel_regularizer=l2(l2_reg))
    model.add(L0)

    # Add subsequent layers
    for layer_idx, nodes in enumerate(layers[1:]):
        model.add(Dense(units=nodes, activation="relu", kernel_regularizer=l2(l2_reg)))

    # Add the output layer
    model.add(Dense(units=3, activation="softmax"))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the target labels
y_encoded = to_categorical(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Set hyperparameters
learning_rate = 0.01
layers = [10, 5]  # Example layer configuration
dropout_rate = 0.2
batch_size = 32
l2_reg = 0.01
ishape = X_train.shape[1]

# Build and train the model
model = build_simple_model(learning_rate, layers, dropout_rate, batch_size, l2_reg, ishape)
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Get class names from the Iris dataset
class_names = iris.target_names

# Open file to write predictions
with open('predictions_report.txt', 'w') as file:
    file.write(f"{'Predicted':<15} | {'Expected':<15}\n")
    file.write(f"{'-'*15} | {'-'*15}\n")
    
    for pred, true in zip(predicted_classes, true_classes):
        file.write(f"{class_names[pred]:<15} | {class_names[true]:<15}\n")

# Print classification report
print(classification_report(true_classes, predicted_classes, target_names=iris.target_names))