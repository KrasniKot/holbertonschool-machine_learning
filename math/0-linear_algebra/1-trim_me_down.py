matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]  # Matrix being evaluated

# Retrieving middle colums (3th and 4th)
the_middle = [row[2:4] for row in matrix]

# Printing middle columns
print("The middle columns of the matrix are: {}".format(the_middle))
