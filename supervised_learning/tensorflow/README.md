# Tensorflow

## Tasks:

### 0. Placeholders:
Write the function `def create_placeholders(nx, classes)`: that returns two placeholders, `x` and `y`, for the neural network:
* `nx`: the number of feature columns in our data
* `classes`: the number of classes in our classifier
  * Returns: placeholders named `x` and `y`, respectively
  * `x` is the placeholder for the input data to the neural network
  * `y` is the placeholder for the one-hot labels for the input data
