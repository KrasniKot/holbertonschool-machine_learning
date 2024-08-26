import numpy as np
import GPyOpt
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

bsm = __import__("6-model").build_simple_model

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

def build_and_train_model(params):
    """ Builds and trains the model """

    lrate, u1, u2, dpout, bsize, l2_reg = params[0]
    print(f"Evaluating parameters:\n\tLearning rate: {lrate}\n\tUnits 1: {u1}\n\tUnits 2: {u2}\n\\")

    model = bsm(
        learning_rate=lrate,
        layers=[u1, u2],
        dropout_rate=dpout,
        batch_size=bsize,  # You can optimize this too if you want
        l2_reg=l2_reg,
        ishape=X_train.shape[1]
    )
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return the best validation loss
    return min(history.history['val_loss'])

# Define the bounds of the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
    {'name': 'layer1_units', 'type': 'discrete', 'domain': (8, 16, 32, 64)},
    {'name': 'layer2_units', 'type': 'discrete', 'domain': (8, 16, 32, 64)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': tuple(range(10, 400))},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)}
]

# Create the Bayesian Optimization object
optimizer = GPyOpt.methods.BayesianOptimization(
    f=build_and_train_model,   # The objective function to minimize
    domain=bounds,             # The bounds on each hyperparameter
    acquisition_type='EI',     # Expected Improvement
    maximize=False             # Minimize the objective function
)

# Run the optimization
optimizer.run_optimization(max_iter=50)

# Save the optimization results
optimizer.plot_convergence()
optimizer.save_report('bayes_opt.txt')

# Print the best parameters
print(f"Best hyperparameters: {optimizer.x_opt}")
