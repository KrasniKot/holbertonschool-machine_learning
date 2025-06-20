""" This module contains two()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """ Plots x ↦ y1 and x ↦ y2 as line graphs """
    # === 1. Model radioactive decay using exponential decay functions for two different decay processes ===
    x  = np.arange(0, 21000, 1000)  # Create an array from 0 to 21000: [0. 1000. 2000, ..., 20000]
    r  = np.log(0.5)                # Natural log of 0.5, represents decay constant
    t1 = 5730                       # Half-life of isotope 1 (Carbon-14)
    t2 = 1600                       # Half-life of isotope 2 (RA-226)
    y1 = np.exp((r / t1) * x)       # Decay curve for isotope 1
    y2 = np.exp((r / t2) * x)       # Decay curve for isotope 2
    # ====== 

    # === 2. Plot the exponential decay ===
    plt.figure(figsize=(6.4, 4.8))  # Create a new figure

    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    plt.plot(x, y1, c="red", linestyle="dashed", label="C-14")
    plt.plot(x, y2, c="green", label="Ra-226")

    plt.legend()

    plt.savefig("plots/3-twoplots.png")
    plt.show()
    # ======


if __name__ == '__main__':
    two()
