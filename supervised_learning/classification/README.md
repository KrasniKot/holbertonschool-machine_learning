# Classification:

## Tasks:

### Task 0. Neuron:
Write a class `Neuron` that defines a single neuron performing binary classification:

* class constructor: `def __init__(self, nx):`  
  * `nx` is the number of input features to the neuron  
    * If `nx` is not an integer, raise a `TypeError `with the exception: `nx must be an integer`  
    * If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be a positive integer`  
  * All exceptions should be raised in the order listed above  
* Public instance attributes:  
  * `W`: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.  
  * `b`: The bias for the neuron. Upon instantiation, it should be initialized to 0.  
  * `A`: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.  

### 1. Privatize Neuron:
* Based on task 0, privatize the instance attributes `W`, `b` and `A`.
* Make getters for each one of them.

### 2. Neuron Forward Propagation:
* Add the public method `def forward_prop(self, X):`  
    * Calculates the forward propagation of the neuron  
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data  
    * `nx` is the number of input features to the neuron  
    * `m` is the number of examples  
  * Updates the private attribute `__A`  
  * The neuron should use a sigmoid activation function  
  * Returns the private attribute `__A`  

### 3. Neuron Cost:
* Add the public method `def cost(self, Y, A):`
	* Calculates the cost of the model using logistic regression
	* `Y` is a numpy.ndarray with shape (`1`, `m`) that contains the correct labels for the input data
	* `A` is a numpy.ndarray with shape (`1`, `m`) containing the activated output of the neuron for each example
	* To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
	* Returns the cost

### 4. Evaluate Neuron:
* Add the public method `def evaluate(self, X, Y):`
  * Evaluates the neuron’s predictions
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * Returns the neuron’s prediction and the cost of the network, respectively
    * The prediction should be a `numpy.ndarray` with shape (`1`, `m`) containing the predicted labels for each example
    * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

### 5. Neuron Gradient Descent:
* Add the public method `def gradient_descent(self, X, Y, A, alpha=0.05):`
  * Calculates one pass of gradient descent on the neuron
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `A` is a `numpy.ndarray` with shape (`1`, `m`) containing the activated output of the neuron for each example
  * `alpha` is the learning rate
  * Updates the private attributes `__W` and `__b`

### 6. Train Neuron:
* Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05):`
  * Trains the neuron
    * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
  * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
    * `iterations` is the number of iterations to train over
    * if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
  * if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  * `alpha` is the learning rate
    * if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    * if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  * All exceptions should be raised in the order listed above
  * Updates the private attributes `__W`, `__b`, and `__A`
  * You are allowed to use one loop
  * Returns the evaluation of the training data after `iterations` of training have occurred

### 7. Upgrade Train Neuron:
* Update the public method train to `def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):`
  * Trains the neuron by updating the private attributes `__W`, `__b`, and `__A`
  * `X `is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `iterations` is the number of iterations to train over
    * if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
    * if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  * `alpha` is the learning rate
    * if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    * if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  * `verbose` is a boolean that defines whether or not to print information about the training. If `True`, print `Cost after {iteration} iterations: {cost}` every `step` iterations:
    * Include data from the 0th and last iteration
  * `graph` is a boolean that defines whether or not to graph information about the training once the training has completed. If `True`:
    * Plot the training data every `step` iterations as a blue line
    * Label the `x-axis` as iteration
    * Label the `y-axis` as cost
    * Title the plot `Training Cost`
  * Include data from the 0th and last iteration
    * Only if either `verbose` or `graph` are `True`:
    * if `step` is not an integer, raise a `TypeError` with the exception `step must be an integer`
  * if `step` is not positive or is greater than `iterations`, raise a `ValueError` with the exception `step must be positive and <= iterations`
  * All exceptions should be raised in the order listed above
  * The 0th iteration should represent the state of the neuron before any training has occurred
  * You are allowed to use one loop
  * You can use `import matplotlib.pyplot as plt`
  * Returns: the evaluation of the training data after `iterations` of training have occurred

### 8. NeuralNetwork:
Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification:
  * class constructor: `def __init__(self, nx, nodes):`
    * `nx` is the number of input features
      * If `nx` is not an integer, raise a TypeError with the exception: `nx` must be an integer
      * If `nx` is less than 1, raise a ValueError with the exception: `nx` must be a positive integer
    * `nodes` is the number of nodes found in the hidden layer
      * If `nodes` is not an integer, raise a `TypeError` with the exception: `nodes must be an integer`
      * If `nodes` is less than 1, raise a `ValueError` with the exception: `nodes must be a positive integer`
    * All exceptions should be raised in the order listed above
  * Public instance attributes:
    * `W1`: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
    * `b1`: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
    * `A1`: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
    * `W2`: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
    * `b2`: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
    * `A2`: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.

### 9 Privatize NeuralNetwork:
Privatize all instanace attributes and set getters for each one of them.

### 10. NeuralNetwork Forward Propagation:
* Add the public method `def forward_prop(self, X):`
  * Calculates the forward propagation of the neural network
  * `X` is a numpy.ndarray with shape (nx, m) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * Updates the private attributes `__A1` and `__A2`
  * The neurons should use a sigmoid activation function
  * Returns the private attributes `__A1` and `__A2`, respectively

### 11. NeuralNetwork Cost:
* Add the public method `def cost(self, Y, A):`
  * Calculates the cost of the model using logistic regression
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `A` is a `numpy.ndarray` with shape (`1`, `m`) containing the activated output of the neuron for each example
  * To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
  * Returns the cost

### 12. Evaluate NeuralNetwork:
* Add the public method `def evaluate(self, X, Y):`
  * Evaluates the neural network’s predictions
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * Returns the neuron’s prediction and the cost of the network, respectively
    * The prediction should be a `numpy.ndarray` with shape (`1`, `m`) containing the predicted labels for each example
    * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

### 13. NeuralNetwork Gradient Descent:
* Add the public method `def gradient_descent(self, X, Y, A1, A2, alpha=0.05):`
  * Calculates one pass of gradient descent on the neural network
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `A1` is the output of the hidden layer
  * `A2` is the predicted output
  * `alpha` is the learning rate
  * Updates the private attributes `__W1`, `__b1`, `__W2`, and `__b2`

### 13. NeuralNetwork Gradient Descent:
* Add the public method `def gradient_descent(self, X, Y, A1, A2, alpha=0.05):`
  * Calculates one pass of gradient descent on the neural network
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `A1` is the output of the hidden layer
  * `A2` is the predicted output
  * `alpha` is the learning rate
  * Updates the private attributes `__W1`, `__b1`, `__W2`, and `__b2`

### 14. Train NeuralNetwork:
* Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05):`
  * Trains the neural network
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `iterations` is the number of iterations to train over
    * if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
    * if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  * `alpha` is the learning rate
    * if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    * if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  * All exceptions should be raised in the order listed above
  * Updates the private attributes `__W1`, `__b1`, `__A1`, `__W2`, `__b2`, and `__A2`
  * You are allowed to use one loop
  * Returns the evaluation of the training data after iterations of training have occurred

### 15. Upgrade Train NeuralNetwork:
### 16. DeepNeuralNetwork:
* class constructor: `def __init__(self, nx, layers):`
  * `nx` is the number of input features
    * If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be an integer`
    * If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be a positive integer`
  * layers is a list representing the number of nodes in each layer of the network
    * If layers is not a list or an empty list, raise a TypeError with the exception: layers must be a list of positive integers
    * The first value in layers represents the number of nodes in the first layer, …
    * If the elements in layers are not all positive integers, raise a TypeError with the exception layers must be a list of positive integers
  * All exceptions should be raised in the order listed above
  * Sets the public instance attributes:
    * L: The number of layers in the neural network.
    * cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.
    * weights: A dictionary to hold all weights and biased of the network. Upon instantiation:
      * The weights of the network should be initialized using the He et al. method and saved in the weights dictionary using the key W{l} where {l} is the hidden layer the weight belongs to
      * The biases of the network should be initialized to 0’s and saved in the weights dictionary using the key b{l} where {l} is the hidden layer the bias belongs to
    * You are allowed to use one loop

### 17. Privatize DeepNeuralNetwork:
Privatize all instanace attributes and set getters for each one of them.

### 18. DeepNeuralNetwork Forward Propagation:
* Add the public method `def forward_prop(self, X):`
  * Calculates the forward propagation of the neural network
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
  * Updates the private attribute `__cache:`
    * The activated outputs of each layer should be saved in the `__cache` dictionary using the key `A{l}` where `{l}` is the hidden layer the activated output belongs to
    * `X` should be saved to the cache dictionary using the key `A0`
  * All neurons should use a sigmoid activation function
  * You are allowed to use one loop
  * Returns the output of the neural network and the cache, respectively

### 19. DeepNeuralNetwork Cost:
* Add the public method `def cost(self, Y, A):`
  * Calculates the cost of the model using logistic regression
  * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
  * `A` is a `numpy.ndarray` with shape (`1`, `m`) containing the activated output of the neuron for each example
  * To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
  * Returns the cost

### 20. Evaluate DeepNeuralNetwork:
* Add the public method `def evaluate(self, X, Y):`
  * Evaluates the neural network’s predictions
  * `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    * `nx` is the number of input features to the neuron
    * `m` is the number of examples
* `Y` is a `numpy.ndarray` with shape `(1`, `m`) that contains the correct labels for the input data
* Returns the neuron’s prediction and the cost of the network, respectively
    * The prediction should be a `numpy.ndarray` with shape (`1`, `m`) containing the predicted labels for each example
    * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

### 21. DeepNeuralNetwork Gradient Descent:
* Add the public method `def gradient_descent(self, Y, cache, alpha=0.05):`
  * Calculates one pass of gradient descent on the neural network
  * `Y` is a `numpy.ndarray` with shape (`1`, m) that contains the correct labels for the input data
  * `cache` is a dictionary containing all the intermediary values of the network
  * `alpha` is the learning rate
  * Updates the private attribute `__weights`
  * You are allowed to use one loop

### 22. Train DeepNeuralNetwork:
* Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05):`
 * Trains the deep neural network
 * `X` is a numpy.ndarray with shape (`nx`, `m`) that contains the input data
   * `nx` is the number of input features to the neuron
   * `m` is the number of examples
 * `Y` is a `numpy.ndarray` with shape (`1`, `m`) that contains the correct labels for the input data
 * `iterations` is the number of iterations to train over
   * if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
   * if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
 * `alpha` is the learning rate
   * if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
   * if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
 * All exceptions should be raised in the order listed above
 * Updates the private attributes `__weights` and `__cache`
 * You are allowed to use one loop
 * Returns the evaluation of the training data after `iterations` of training have occurred

### 23. Upgrade Train DeepNeuralNetwork:
### 24. One-Hot Encode:
Write a function `def one_hot_encode(Y, classes):` that converts a numeric label vector into a one-hot matrix:
* `Y` is a `numpy.ndarray` with shape (`m`,) containing numeric class labels
* `m` is the number of examples
* `classes` is the maximum number of classes found in `Y`
* Returns: a one-hot encoding of `Y` with shape (`classes`, `m`), or `None` on failure

### 25. One-Hot Decode:
Write a function `def one_hot_decode(one_hot)`: that converts a one-hot matrix into a vector of labels:
* `one_hot` is a one-hot encoded numpy.ndarray with shape (`classes`, `m`)
  * classes is the maximum number of classes
  * m is the number of examples
* Returns: a `numpy.ndarray` with shape (`m`, ) containing the numeric labels for each example, or None on failure

### 26. Persistence is Key:
Update the class DeepNeuralNetwork (based on `23-deep_neural_network.py`):
* Create the instance method `def save(self, filename)`:
  * Saves the instance object to a file in `pickle` format
  * `filename` is the file to which the object should be saved
  * If filename does not have the extension `.pkl`, add it
* Create the static method def `load(filename)`:
  * Loads a pickled `DeepNeuralNetwork` object
  * `filename` is the file from which the object should be loaded
  * Returns: the loaded object, or `None` if `filename` doesn’t exist

### 27. Update DeepNeuralNetwork:
Update the class DeepNeuralNetwork to perform multiclass classification (based on `26-deep_neural_network.py`):
* You will need to update the instance methods `forward_prop`, `cost`, and `evaluate`
* `Y` is now a one-hot numpy.ndarray of shape (`classes`, `m`)

### 28. All the activations:
Update the class `DeepNeuralNetwork` to allow different activation functions (based on ``27-deep_neural_network.py``):

Update the ``__init__`` method to def ``__init__(self, nx, layers, activation='sig'):``
- ``activation`` represents the type of activation function used in the hidden layers
- ``sig`` represents a sigmoid activation
- ``tanh`` represents a tanh activation
  - if ``activation`` is not `sig` or ``tanh``, raise a `ValueError` with the exception: `activation must be 'sig' or 'tanh'`
- Create the private attribute `__activation` and set it to the value of `activation`
- Create a getter for the private attribute `__activation`
- Update the `forward_prop` and `gradient_descent` instance methods to use the `__activation` function in the hidden layers