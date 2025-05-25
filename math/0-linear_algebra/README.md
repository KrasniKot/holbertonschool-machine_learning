# Linear Algebra

## Requirements

- [x] Python 3
- [x] Pip
- [x] Numpy

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
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 2-size_me_please.py 
Shape matrix 1: [2, 2]
Shape matrix 2: [2, 3, 5]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# 
```


### 3. Flip Me Over
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:
- You must return a new matrix
- You can assume that `matrix` is never empty
- You can assume all elements in the same dimension are of the same type/shape

#### Execution
The output should look something like
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 3-flip_me_over.py 
Matrix 1: [[1, 2], [3, 4]]
[(1, 3), (2, 4)]
--------------------------------------------------
Matrix 2: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[(1, 6, 11, 16, 21, 26), (2, 7, 12, 17, 22, 27), (3, 8, 13, 18, 23, 28), (4, 9, 14, 19, 24, 29), (5, 10, 15, 20, 25, 30)]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 4. Line Up
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:
- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list
- If `arr1` and `arr2` are not the same shape, return `None`

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 4-line_up.py 
Array addition (arr1, arr2): [6, 8, 10, 12]
Array 1: [1, 2, 3, 4]
Array 2: [5, 6, 7, 8]
None
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# 
```


### 5. Across The Planes
Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If `mat1` and `mat2` are not the same shape, return `None`

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 5-across_the_planes.py 
Mat1 + Mat2: [[6, 8], [10, 12]]
Mat1: [[1, 2], [3, 4]]
Mat2: [[5, 6], [7, 8]]
None
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# 
```


### 6. Howdy Partner
Write a function `def cat_arrays(arr1, arr2):` that concatenates two arrays:
- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 6-howdy_partner.py 
---------- Two arrays ----------
Array 1: [1, 2, 3, 4, 5]
Array 2: [6, 7, 8]

Concatenation... [1, 2, 3, 4, 5, 6, 7, 8]
---------- Two empty arrays ----------
Array 1: []
Array 2: []

Concatenation... []
---------- One empty array ----------
Array 1: [1, 2, 3, 4, 5]
Array 2: []

Concatenation... [1, 2, 3, 4, 5]
```


### 7. Gettin’ Cozy
Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be concatenated, return `None`

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 7-gettin_cozy.py 
Matrix1 + Matrix2, along first axis: [[1, 2], [3, 4], [5, 6]]
Matrix1 + Matrix3, along second axis: [[1, 2, 7], [3, 4, 8]]
```


### 8. Ridin’ Bareback
Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:
- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be multiplied, return `None`

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python3 8-ridin_bareback.py 
First test: [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
Square Matrix Test (3x3): [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
Non-Square Matrices Test (3x2 * 2x3): [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
Rectangular Matrices Test (2x3 * 3x1): [[50], [122]]
Zero Matrix Test: [[0, 0], [0, 0], [0, 0]]
Single Element Matrix Test: [[6]]
Non-Commutative Test A * B: [[19, 22], [43, 50]]
Non-Commutative Test B * A: [[23, 34], [31, 46]]
Incompatible Dimensions (2x3 * 2x2): None
Incompatible Dimensions (3x3 * 1x2): None
Empty Matrix Test: None
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# 
```


### 9. Let The Butcher Slice It
Complete the following source code:
- `mat1` should be the middle two rows of `matrix`
- `mat2` should be the middle two columns of `matrix`
- `mat3` should be the bottom-right, square, 3x3 matrix of `matrix`
- You are not allowed to use any loops or conditional statements
- Your program should be exactly 10 lines

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 9-let_the_butcher_slice_it.py 
The middle two rows of the matrix are:
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]

The middle two columns of the matrix are:
[[ 3  4]
 [ 9 10]
 [15 16]
 [21 22]]

The bottom-right, square, 3x3 matrix is:
[[10 11 12]
 [16 17 18]
 [22 23 24]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 10. I'll Use My Scale
Write a function `def np_shape(matrix):` that calculates the shape of a `numpy.ndarray`:
- You are not allowed to use any loops or conditional statements
- You are not allowed to use `try/except` statements
- The shape should be returned as a tuple of integers

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 10-ill_use_my_scale.py 
mat1 shape: (6,)
mat2 shape: (0,)
mat3 shape: (2, 2, 5)
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 11. The Western Exchange
Write a function `def np_transpose(matrix):` that transposes matrix:
- You can assume that `matrix` can be interpreted as a `numpy.ndarray`
- You are not allowed to use any loops or conditional statements
- You must return a new `numpy.ndarray`

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 11-the_western_exchange.py 
----------------------------------------
Matrix 1:
[1 2 3 4 5 6]
Transpose of Matrix 1:
[1 2 3 4 5 6]
----------------------------------------
Matrix 2:
[]
Transpose of Matrix 2:
[]
----------------------------------------
Matrix 3:
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]]
Transpose of Matrix 3:
[[[ 1 11]
  [ 6 16]]

 [[ 2 12]
  [ 7 17]]

 [[ 3 13]
  [ 8 18]]

 [[ 4 14]
  [ 9 19]]

 [[ 5 15]
  [10 20]]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 12. Bracing The Elements
Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:
- You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarrays`
- You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
- You are not allowed to use any loops or conditional statements
- You can assume that `mat1` and `mat2` are never empty

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 12-bracin_the_elements.py 
Matrix 1:
 [[11 22 33]
 [44 55 66]]
Matrix 2:
 [[1 2 3]
 [4 5 6]]
---------------------------------------- matrix to matrix operations ----------------------------------------
Add (Element-wise Addition):
 [[12 24 36]
 [48 60 72]]
Sub (Element-wise Subtraction):
 [[10 20 30]
 [40 50 60]]
Mul (Hadamard Product):
 [[ 11  44  99]
 [176 275 396]]
Div (Element-wise Division):
 [[11. 11. 11.]
 [11. 11. 11.]]
---------------------------------------- matrix to scalar (2) operations ----------------------------------------
Add (Element-wise Addition with scalar):
 [[13 24 35]
 [46 57 68]]
Sub (Element-wise Subtraction with scalar):
 [[ 9 20 31]
 [42 53 64]]
Mul (Hadamard Product with scalar):
 [[ 22  44  66]
 [ 88 110 132]]
Div (Element-wise Division with scalar):
 [[ 5.5 11.  16.5]
 [22.  27.5 33. ]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 13. Cat's Got Your Tongue:
Write a function `def np_cat(mat1, mat2, axis=0)` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
- You must return a new `numpy.ndarray`
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty

#### Execution
```
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 13-cats_got_your_tongue.py 
Concatenation mat1 and mat2 (axis 0):
 [[11 22 33]
 [44 55 66]
 [ 1  2  3]
 [ 4  5  6]]

Concatenation mat1 and mat2 (axis 1):
 [[11 22 33  1  2  3]
 [44 55 66  4  5  6]]

Concatenation mat1 and mat1 (axis 1):
 [[11 22 33  7]
 [44 55 66  8]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 14. Saddle Up:
Write a function def `np_matmul(mat1, mat2):` that performs matrix multiplication:
- You can assume that `mat1` and `mat2` are `numpy.ndarray`s
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty

#### Execution:
```
oot@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra# python 14-saddle_up.py 
matrix 1 @ matrix 2:
 [[ 330  396  462]
 [ 726  891 1056]]

matrix 1 @ matrix 3:
 [[ 550]
 [1342]]
root@ffbdd98be718:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 15. Slice Like A Ninja
Write a function `def np_slice(matrix, axes={}):` that slices a matrix along specific axes:
- You can assume that `matrix` is a `numpy.ndarray`
- You must return a new `numpy.ndarray`
- `axes` is a dictionary where the key is an axis to slice along and the `value` is a tuple representing the slice to make along that axis
- You can assume that axes represents a valid slice

#### Execution
```
root@f19c3905f049:~/holbertonschool-machine_learning/math/0-linear_algebra# python 100-slice_like_a_ninja.py 
First matrix:
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
Matrix slice:
[[2 3]
 [7 8]]

------------------------------
Second matrix:
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]

 [[21 22 23 24 25]
  [26 27 28 29 30]]]
Matrix slice:
[[[ 5  3  1]
  [10  8  6]]

 [[15 13 11]
  [20 18 16]]]
root@f19c3905f049:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 16. The Whole Barn
Write a function `def add_matrices(mat1, mat2):` that adds two matrices:
- You can assume that `mat1` and `mat2` are matrices containing `ints/floats`
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If matrices are not the same shape, return `None`
- You can assume that `mat1` and `mat2` will never be empty

#### Execution
```
root@6b792e227861:~/holbertonschool-machine_learning/math/0-linear_algebra# python 101-the_whole_barn.py 
Addition equals:
[5, 7, 9]

Addition equals:
[[6, 8], [10, 12]]
[2, 3, 2, 4]

Addition equals:
[[[[12, 14, 16, 18], [20, 22, 24, 26]], [[28, 120, 122, 124], [126, 128, 130, 132]], [[134, 136, 138, 140], [142, 144, 146, 148]]], [[[150, 152, 154, 156], [158, 160, 162, 164]], [[166, 168, 170, 172], [174, 176, 178, 180]], [[182, 184, 186, 188], [190, 192, 194, 196]]]]

Addition equals:
None
root@6b792e227861:~/holbertonschool-machine_learning/math/0-linear_algebra#
```


### 17. Squashed Like Sardines
- Write a function `def cat_matrices(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:
- You can assume that `mat1` and `mat2` are matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If you cannot concatenate the matrices, return `None`
- You can assume that `mat1` and `mat2` are never empty

