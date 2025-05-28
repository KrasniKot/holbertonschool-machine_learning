""" This module contains scatter(),

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """ Plots x ↦ y as a scatter plot """
    
    # Define the mean and covariance matrix for the 2D normal distribution
    mean = [69, 0]             # Mean height = 69 inches, mean weight deviation = 0
    cov  = [[15, 8], [8, 15]]  # Covariance matrix showing height and weight variability

    # Set random seed for reproducibility
    np.random.seed(5)

    # Generate 2000 (height, weight deviation) samples from a multivariate normal distribution
    x, y = np.random.multivariate_normal(mean, cov, 2000).T

    # Shift all weights by 180 lbs to simulate realistic weight values
    y += 180

    # Create a new figure
    plt.figure(figsize=(6.4, 4.8))

    # Set the plot title and axis labels
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    
    plt.scatter(x, y, c="magenta")      # Create the scatter plot with magenta-colored points
    plt.savefig("plots/1-scatter.png")  # Save the plot image to a file
    plt.show()                          # Display the plot


if __name__ == '__main__':
    scatter()
