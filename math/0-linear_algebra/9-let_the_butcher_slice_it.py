""" Perform some slicing operations on numpy arrays """
import numpy as np


# Declare the matrix
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])

# Slicing operations
mat1 = matrix[1:3]
mat2 = matrix[:, 2:4]
mat3 = matrix[-3:, -3:]

# Print the slices
print(f"The middle two rows of the matrix are:\n{mat1}\n")
print(f"The middle two columns of the matrix are:\n{mat2}\n")
print(f"The bottom-right, square, 3x3 matrix is:\n{mat3}")
