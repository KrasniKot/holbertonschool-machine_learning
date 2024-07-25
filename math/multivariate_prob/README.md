# Multivariate Probability

## Tasks

### 0. Mean and Covariance:
Write a function ``def mean_cov(X):`` that calculates the mean and covariance of a data set:

- ``X`` is a ``numpy.ndarray`` of shape ``(n, d)`` containing the data set:
- ``n`` is the number of data points
- ``d`` is the number of dimensions in each data point
- If ``X`` is not a 2D ``numpy.ndarray``, raise a ``TypeError`` with the message ``X must be a 2D numpy.ndarray``
- If ``n`` is less than 2, raise a ValueError with the message ``X must contain multiple data points``
- Returns: ``mean``, ``cov``:
- ``mean`` is a ``numpy.ndarray`` of shape ``(1, d)`` containing the mean of the data set
- ``cov`` is a ``numpy.ndarray`` of shape ``(d, d)`` containing the covariance matrix of the data set
- You are not allowed to use the function ``numpy.cov``

### 1. Correlation:
Write a function ``def correlation(C):`` that calculates a correlation matrix:

- ``C`` is a ``numpy.ndarray`` of shape ``(d, d)`` containing a covariance matrix
- ``d`` is the number of dimensions
- If ``C`` is not a ``numpy.ndarray``, raise a ``TypeError`` with the message ``C must be a numpy.ndarray``
- If ``C`` does not have shape ``(d, d)``, raise a ``ValueError`` with the message ``C must be a 2D square matrix``
- Returns a ``numpy.ndarray`` of shape ``(d, d)`` containing the correlation matrix

### 2. Initialize:
### 3. PDF:
