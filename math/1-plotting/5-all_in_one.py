""" This module contains all_in_one()

    requires:
        - numpy,
        - matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """ Slots all 5 previous graphs in one figure """
    # === First plot data ===
    y0 = np.arange(0, 11) ** 3
    # ======

    # === Second plot data ===
    np.random.seed(5)

    mean   = [69, 0]
    cov    = [[15, 8], [8, 15]]
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1    += 180
    # ======

    # === Third plot data ===
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    # ======

    # === Fourth plot data === 
    x3  = np.arange(0, 21000, 1000)
    r3  = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    # ======

    # === Fifth plot data ===
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    # ======

    fig = plt.figure()
    fig.suptitle("All in One")

    # === Plot 1 ===
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(y0, "red")
    ax1.set_xlim((0, 10))
    ax1.set_yticks([0, 500, 1000])
    # ======

    # === Plot 2 ===
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title("Men's Height vs Weight", fontsize="x-small")
    ax2.set_xlabel("Height (in)", fontsize="x-small")
    ax2.set_ylabel("Weight (lbs)", fontsize="x-small")
    ax2.scatter(x1, y1, c="magenta")
    # ======

    # === Plot 3 ===
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title("Exponential Decay of C-14", fontsize="x-small")
    ax3.set_xlabel("Time (years)", fontsize="x-small")
    ax3.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax3.set_yscale("log")
    ax3.set_xlim((0, 28650))
    ax3.plot(x2, y2)
    # ======

    # === Plot 4 ===
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title("Exponential Decay of Radioactive Elements", fontsize="x-small")
    ax4.set_xlabel("Time (years)", fontsize="x-small")
    ax4.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax4.set_xlim((0, 20000))
    ax4.set_ylim((0, 1))
    ax4.plot(x3, y31, "red", linestyle="dashed", label="C-14")
    ax4.plot(x3, y32, "green", label="Ra-226")
    ax4.legend()
    # ======

    # === Plot 5 ===
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.set_title("Project A", fontsize="x-small")
    ax5.set_xlabel("Grades", fontsize="x-small")
    ax5.set_ylabel("Number of Students", fontsize="x-small")
    ax5.set_xlim((0, 100))
    ax5.set_xticks(range(0, 101, 10))
    ax5.set_ylim((0, 30))
    ax5.hist(student_grades, bins=[10 * i for i in range(0, 11)], edgecolor="black")
    # ======


    plt.tight_layout()

    plt.savefig("plots/5-fiveplots.png")
    plt.show()


if __name__ == '__main__':
    all_in_one()