arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]  # Array to be sliced

# Array slicing
arr1 = arr[0:2]  # Covering the first two numbers (9, 8)
arr2 = arr[-5:]  # Covering the last five numbers (9, 4, 1, 0, 3)
arr3 = arr[1:6]  # Covering the 2nd through 6th numbers (8, 2, 3, 9, 4)

# Slice printing
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
