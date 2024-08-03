# Clustering

## Tasks:

### 0. Initialize K-means:
Write a function ``def initialize(X, k):`` that initializes cluster centroids for K-means:

- ``X`` is a numpy.ndarray of shape ``(n, d)`` containing the dataset that will be used for K-means clustering
  - ``n`` is the number of data points
  - ``d`` is the number of dimensions for each data point
- ``k`` is a positive integer containing the number of clusters
- The cluster centroids should be initialized with a multivariate uniform distribution along each dimension in ``d``:
  - The minimum values for the distribution should be the minimum values of ``X`` along each dimension in ``d``
  - The maximum values for the distribution should be the maximum values of ``X`` along each dimension in ``d``
  - You should use ``numpy.random.uniform`` exactly once
- You are not allowed to use any loops
- Returns: a ``numpy.ndarray`` of shape ``(k, d) ``containing the initialized centroids for each cluster, or ``None`` on failure

### 1. K-means:
Write a function def kmeans(X, k, iterations=1000): that performs K-means on a dataset:

- X is a numpy.ndarray of shape (n, d) containing the dataset
- n is the number of data points
- d is the number of dimensions for each data point
- k is a positive integer containing the number of clusters
- iterations is a positive integer containing the maximum number of iterations that should be performed
- If no change in the cluster centroids occurs between iterations, your function should return
- Initialize the cluster centroids using a multivariate uniform distribution (based on0-initialize.py)
- If a cluster contains no data points during the update step, reinitialize its centroid
- You should use numpy.random.uniform exactly twice
- You may use at most 2 loops
- Returns: C, clss, or None, None on failure
- C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
- clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to

### 2. Variance:
Write a function ``def variance(X, C):`` that calculates the total intra-cluster variance for a data set:

- ``X`` is a ``numpy.ndarray`` of shape ``(n, d)`` containing the data set
- ``C`` is a ``numpy.ndarray`` of shape ``(k, d)`` containing the centroid means for each cluster
- You are not allowed to use any loops
- Returns: ``var``, or ``None`` on failure
  - ``var`` is the total variance

### 3. Optimize k:
Write a function ``def optimum_k(X, kmin=1, kmax=None, iterations=1000):`` that tests for the optimum number of clusters by variance:

- ``X`` is ``a numpy.ndarray`` of shape ``(n, d)`` containing the data set
- ``kmin`` is a positive integer containing the minimum number of clusters to check for (inclusive)
- ``kmax`` is a positive integer containing the maximum number of clusters to check for (inclusive)
- ``iterations`` is a positive integer containing the maximum number of iterations for K-means
- This function should analyze at least 2 different cluster sizes
- You should use:
  - ``kmeans = __import__('1-kmeans').kmeans``
  - ``variance = __import__('2-variance').variance``
- You may use at most 2 loops
- Returns: ``results``, ``d_vars``, or ``None``, ``None`` on failure
  - ``results`` is a list containing the outputs of K-means for each cluster size
  - ``d_vars`` is a list containing the difference in variance from the smallest cluster size for each cluster size`

### 4. Initialize GMM:
Write a function ``def initialize(X, k):`` that initializes variables for a Gaussian Mixture Model:

- ``X`` is a ``numpy.ndarray`` of shape ``(n, d)`` containing the data set
- ``k`` is a positive integer containing the number of clusters
- You are not allowed to use any loops
- Returns: ``pi``, ``m``, ``S``, or ``None``, ``None``, ``None`` on failure
  - ``pi`` is a ``numpy.ndarray`` of shape ``(k,)`` containing the priors for each cluster, initialized evenly
  - ``m`` is a ``numpy.ndarray`` of shape ``(k, d)`` containing the centroid means for each cluster, initialized with K-means
  - ``S`` is a ``numpy.ndarray`` of shape ``(k, d, d)`` containing the covariance matrices for each cluster, initialized as identity matrices
- You should use ``kmeans = __import__('1-kmeans').kmeans``

### 5. PDF:
### 6. Expectation:
### 7. Maximization:
### 8. EM:
### 9. BIC:
### 10. Hello, sklearn!:
### 11. GMM:
### 12. Agglomerative:
