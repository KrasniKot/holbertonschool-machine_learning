# Optimization

## Tasks:

### 0. Normalization Constants:
Write the function `def normalization_constants(X):` that calculates the normalization (standardization) constants of a matrix:
* `X` is the `numpy.ndarray` of shape (`m`, `nx`) to normalize
  * `m` is the number of data points
  * `nx` is the number of features
* Returns: the mean and standard deviation of each feature, respectively

### 1. Normalize:
Write the function def normalize(`X`, `m`, `s`): that normalizes (standardizes) a matrix:
* `X` is the `numpy.ndarray` of shape (`d`, `nx`) to normalize
  * `d` is the number of data points
  * `nx` is the number of features
* `m` is a `numpy.ndarray` of shape (`nx`,) that contains the mean of all features of `X`
* `s` is a `numpy.ndarray` of shape (`nx`,) that contains the standard deviation of all features of `X`
* Returns: The normalized `X` matrix

### 2. Shuffle Data:
Write the function `def shuffle_data(X, Y):` that shuffles the data points in two matrices the same way:
* `X` is the first `numpy.ndarray` of shape (`m`, `nx`) to shuffle
  * `m` is the number of data points
  * `nx` is the number of features in `X`
* `Y` is the second `numpy.ndarray` of shape (`m`, `ny`) to shuffle
  * `m` is the same number of data points as in `X`
  * `ny` is the number of features in `Y`
* Returns: the shuffled `X` and `Y matrices

### 3. Mini-Batch:
Write the function `def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):` that trains a loaded neural network model using mini-batch gradient descent:

* `X_train` is a `numpy.ndarray` of shape `(m, 784)` containing the training data
  * `m` is the number of data points
  * `784` is the number of input features
* `Y_train` is a one-hot `numpy.ndarray` of shape `(m, 10)` containing the training labels
  * `10` is the number of classes the model should classify
* `X_valid` is a `numpy.ndarray` of shape `(m, 784)` containing the validation data
* `Y_valid` is a one-hot `numpy.ndarray` of shape `(m, 10)` containing the validation labels
* `batch_size` is the number of data points in a batch
* `epochs` is the number of times the training should pass through the whole dataset
* `load_path` is the path from which to load the model
* `save_path` is the path to where the model should be saved after training
* Returns: the path where the model was saved
* Your training function should allow for a smaller final batch (a.k.a. use the entire training set)
* 1) import meta graph and restore session
* 2) Get the following tensors and ops from the collection restored
  * `x` is a placeholder for the input data
  * `y` is a placeholder for the labels
  * `accuracy` is an op to calculate the accuracy of the model
  * `loss` is an op to calculate the cost of the model
  * `train_op` is an op to perform one pass of gradient descent on the model
* 3) loop over epochs:
  * shuffle data
  * loop over the batches:
  * get `X_batch` and `Y_batch` from data
  * train your model
* 4) Save session
* You should use `shuffle_data = __import__('2-shuffle_data').shuffle_data`
* Before the first epoch and after every subsequent epoch, the following should be printed:
  * `After {epoch} epochs`: where `{epoch}` is the current epoch
  * `\tTraining Cost: {train_cost}` where `{train_cost}` is the cost of the model on the entire training set
  * `\tTraining Accuracy: {train_accuracy}` where `{train_accuracy}` is the accuracy of the model on the entire training set
  * `\tValidation Cost: {valid_cost}` where `{valid_cost}` is the cost of the model on the entire validation set
  * `\tValidation Accuracy: {valid_accuracy}` where `{valid_accuracy}` is the accuracy of the model on the entire validation set
* After every 100 steps gradient descent within an epoch, the following should be printed:
  * `\tStep {step_number}:` where `{step_number}` is the number of times gradient descent has been run in the current epoch
  * `\t\tCost: {step_cost}` where `{step_cost}` is the cost of the model on the current mini-batch
  * `\t\tAccuracy: {step_accuracy} where `{step_accuracy}` is the accuracy of the model on the current mini-batch
  * Advice: the function `range` can help you to handle this loop inside your dataset by using `batch_size` as step value

### 4. Moving Average:
### 5. Momentum:
### 6. Momentum Upgraded:
### 7. RMSProp:
### 8. RMSProp Upgraded:
### 9. Adam:
### 10. Adam Upgraded:
### 11. Learning Rate Decay:
### 12. Learning Rate Decay Upgraded:
### 13. Batch Normalization:
### 14. Batch Normalization Upgraded:
