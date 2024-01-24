# Error Analysis

## Tasks:

### 0. Create Confusion:
Write the function def `create_confusion_matrix(labels, logits):` that creates a confusion matrix:
* `labels` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the correct labels for each data point
* `m` is the number of data points
* `classes` is the number of classes
* `logits` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the predicted labels
* Returns: a confusion `numpy.ndarray` of shape `(classes, classes)` with row indices representing the correct labels and column indices representing the predicted labels

### 1. Sensitivity:
Write the function `def sensitivity(confusion):` that calculates the sensitivity for each class in a confusion matrix:
* `confusion` is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
* `classes` is the number of classes
* Returns: a `numpy.ndarray` of shape `(classes,)` containing the sensitivity of each class

