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
