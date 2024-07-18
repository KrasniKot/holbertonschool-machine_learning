import numpy as np

A = np.array([
    [-2, -4, 2],
    [-2, 1, 2],
    [4, 2, 5]
])

A_power_10 = np.linalg.matrix_power(A, 10)
print(A_power_10)
