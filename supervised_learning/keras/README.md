# Keras

## Tasks:

### 0. Sequential:
Write a function `def build_model(nx, layers, activations, lambtha, keep_prob):` that builds a neural network with the Keras library:

* ``nx`` is the number of input features to the network
* ``layers`` is a list containing the number of nodes in each layer of the network
* ``activations`` is a list containing the activation functions used for each layer of the network
* ``lambtha`` is the ``L2`` regularization parameter
* ``keep_prob`` is the probability that a node will be kept for dropout

You are not allowed to use the Input class
Returns: the keras model

### 1. Input:
Write a function ``def build_model(nx, layers, activations, lambtha, keep_prob):`` that builds a neural network with the Keras library:
- ``nx`` is the number of input features to the network
- ``layers`` is a list containing the number of nodes in each layer of the network
- ``activations`` is a list containing the activation functions used for each layer of the network
- ``lambtha`` is the L2 regularization parameter
- ``keep_prob`` is the probability that a node will be kept for dropout

You are not allowed to use the Sequential class
Returns: the keras model

### 2. Optimize:
Write a function ``def optimize_model(network, alpha, beta1, beta2):`` that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:
- ``network`` is the model to optimize
- ``alpha`` is the learning rate
- ``beta1`` is the first Adam optimization parameter
- ``beta2`` is the second Adam optimization parameter

Returns: None

### 3. One Hot:
Write a function ``def one_hot(labels, classes=None):`` that converts a label vector into a one-hot matrix:
- The last dimension of the one-hot matrix must be the number of classes
Returns: the one-hot matrix

### 4. Train:
Write a function ``def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):`` that trains a model using mini-batch gradient descent:
- ``network`` is the model to train
- ``data`` is a ``numpy.ndarray`` of shape ``(m, nx)`` containing the input data
- ``labels`` is a one-hot ``numpy.ndarray`` of shape ``(m, classes)`` containing the labels of data
- ``batch_size`` is the size of the batch used for mini-batch gradient descent
- ``epochs`` is the number of passes through data for mini-batch gradient descent
- ``verbose`` is a boolean that determines if output should be printed during training
- ``shuffle`` is a boolean that determines whether to shuffle the batches every epoch. Normally, it is a good idea to shuffle, but for reproducibility, we have chosen to set the default to ``False``.

Returns: the History object generated after training the model

### 5. Validate:
Based on 4-train.py, update the function ``def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):`` to also analyze validaiton data:

- ``validation_data`` is the data to validate the model with, if not ``None``

### 6. Early Stopping:
Based on 5-train.py, update the function ``def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):`` to also train the model using early stopping:

- ``early_stopping`` is a boolean that indicates whether early stopping should be used
- ``early`` stopping should only be performed if validation_data exists
- ``early`` stopping should be based on validation loss
- ``patience`` is the patience used for early stopping

### 7. Learning Rate Decay:
Based on 6-train.py, update the function ``def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):`` to also train the model with learning rate decay:
- `learning_rate_decay` is a boolean that indicates whether learning rate decay should be used
  - learning rate decay should only be performed if validation_data exists
  - the decay should be performed using inverse time decay
  - the learning rate should decay in a stepwise fashion after each epoch
  - each time the learning rate updates, Keras should print a message
- ``alpha`` is the initial learning rate
- ``decay_rate`` is the decay rate

### 8. Save Only the Best:
Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model:

save_best is a boolean indicating whether to save the model after each epoch if it is the best
a model is considered the best if its validation loss is the lowest that the model has obtained
filepath is the file path where the model should be saved

### 9. Save and Load Model:
### 10. Save and Load Weights:
### 11. Save and Load Configuration:
### 12. Test:
### 13. Predict:
