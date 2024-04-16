#!/usr/bin/env python3
""" This module contains frequency()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ Plots a histogram of student scores for a project """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.hist(student_grades, bins=[10 * i for i in range(0, 11)],
             edgecolor="black")
    plt.xticks(range(0, 110, 10))

#    plt.savefig("histplot.png")
    plt.show()
