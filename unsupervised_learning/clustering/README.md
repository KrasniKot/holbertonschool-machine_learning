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
### 2. Variance:
### 3. Optimize k:
### 4. Initialize GMM:
### 5. PDF:
### 6. Expectation:
### 7. Maximization:
### 8. EM:
### 9. BIC:
### 10. Hello, sklearn!:
### 11. GMM:
### 12. Agglomerative:
