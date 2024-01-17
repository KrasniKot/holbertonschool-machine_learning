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
