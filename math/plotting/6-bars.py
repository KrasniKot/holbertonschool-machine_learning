#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
people = ["Farrah", "Fred", "Felicia"]

for i, f in enumerate(["Apples", "Bananas", "Oranges", "Peaches"]):
    plt.bar(
        range(len(people)),
        fruit[i],
        label=f,
        color=colors[i],
        bottom=np.sum(fruit[:i], axis=0),
        width=0.5
        )

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.xlabel("Person")
plt.xticks(range(len(people)), people)
plt.yticks(range(0, 81, 10))
plt.legend()

plt.show()
