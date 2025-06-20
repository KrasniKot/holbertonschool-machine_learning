""" This module contains bars()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Plots a stacked bar graph """
    # === Create a random matrix (4, 3) from a gaussian distribution ===
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    # ======

    # === Create the stacked bar plot ===
    plt.figure(figsize=(6.4, 4.8))

    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    people = ["Farrah", "Fred", "Felicia"]

    for i, f in enumerate(["Apples", "Bananas", "Oranges", "Peaches"]):
        plt.bar(range(len(people)), fruit[i], label=f, color=colors[i], bottom=np.sum(fruit[:i], axis=0), width=0.5)

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.xticks(range(len(people)), people)
    plt.yticks(range(0, 81, 10))
    plt.legend()

    plt.savefig("plots/6-stacked.png")
    plt.show()
    # ======


if __name__ == '__main__':
    bars()