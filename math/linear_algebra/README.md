# Linear Algebra

## Requirements

- [x] Python 3

## Tasks

### 0. Slice Me Up
Complete the source code:
- `arr1` should be the first two numbers of `arr`
- `arr2` should be the last five numbers of `arr`
- `arr3` should be the 2nd through 6th numbers of `arr`
- You are not allowed to use any loops or conditional statements

#### Execution
To execute this program run `python3 0-slice_me_up.py`

The output should look something like:
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# python3 0-slice_me_up.py 
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# 
```

### 1. Trim Me Down
Complete the source code
- `the_middle` should be a 2D matrix containing the 3rd and 4th columns of `matrix`
- You are not allowed to use any conditional statements
- You are only allowed to use one `for` loop

#### Execution
The output should look something like:
```
root@ffbdd98be718:~/holbertonschool-machine_learning# python3 math/linear_algebra/1-trim_me_down.py 
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
root@ffbdd98be718:~/holbertonschool-machine_learning# 
```

### 2. Size Me Please
Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:
- You can assume all elements in the same dimension are of the same type/shape
- The shape should be returned as a list of integers

#### Execution
The output should look something like this:
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# python3 2-main.py 
[2, 2]
[2, 3, 5]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# 
```

### 3. Flip Me Over
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:
- You must return a new matrix
- You can assume that `matrix` is never empty
- You can assume all elements in the same dimension are of the same type/shape

#### Execution
The output should look something like
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# python3 3-main.py 
[[1, 2], [3, 4]]
[[1, 3], [2, 4]]
--------------------------------------------------
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/linear_algebra# 
```

### 4. Line Up:
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:
- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list
- If `arr1` and `arr2` are not the same shape, return `None`

### 5. Across The Planes:
Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If `mat1` and `mat2` are not the same shape, return `None`

### 6. Howdy Partner:
Write a function `def cat_arrays(arr1, arr2):` that concatenates two arrays:
- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list

### 7. Gettin’ Cozy:
Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be concatenated, return `None`

### 8. Ridin’ Bareback:
Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be multiplied, return `None`

### 9. Let The Butcher Slice It:
Complete the following source code (fond below):
- `mat1` should be the middle two rows of `matrix`
- `mat2` should be the middle two columns of `matrix`
- `mat3` should be the bottom-right, square, 3x3 matrix of `matrix`
- You are not allowed to use any loops or conditional statements
- Your program should be exactly 10 lines

### 10. I'll Use My Scale:
Write a function `def np_shape(matrix):` that calculates the shape of a `numpy.ndarray`:
- You are not allowed to use any loops or conditional statements
- You are not allowed to use `try/except` statements
- The shape should be returned as a tuple of integers

### 11. The Western Exchange:
Write a function `def np_transpose(matrix):` that transposes matrix:
- You can assume that `matrix` can be interpreted as a `numpy.ndarray`
- You are not allowed to use any loops or conditional statements
- You must return a new `numpy.ndarray`

### 12. Bracing The Elements:
Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:
- You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarrays`
- You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
- You are not allowed to use any loops or conditional statements
- You can assume that `mat1` and `mat2` are never empty

### 13. Cat's Got Your Tongue:
Write a function `def np_cat(mat1, mat2, axis=0)` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
- You must return a new `numpy.ndarray`
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty

### 14. Saddle Up:
Write a function def `np_matmul(mat1, mat2):` that performs matrix multiplication:
- You can assume that `mat1` and `mat2` are `numpy.ndarray`s
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty

### 15. Slice Like A Ninja
Write a function `def np_slice(matrix, axes={}):` that slices a matrix along specific axes:
- You can assume that `matrix` is a `numpy.ndarray`
- You must return a new `numpy.ndarray`
- `axes` is a dictionary where the key is an axis to slice along and the `value` is a tuple representing the slice to make along that axis
- You can assume that axes represents a valid slice

### 16. The Whole Barn:
Write a function `def add_matrices(mat1, mat2):` that adds two matrices:
- You can assume that `mat1` and `mat2` are matrices containing `ints/floats`
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If matrices are not the same shape, return `None`
- You can assume that `mat1` and `mat2` will never be empty

### 17. Squashed Like Sardines:
- Write a function `def cat_matrices(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` are matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If you cannot concatenate the matrices, return `None`
- You can assume that `mat1` and `mat2` are never empty

