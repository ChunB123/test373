import numpy as np
import matplotlib.pyplot as plt


def binary_cross_entropy(p, q):
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


p = .75

q_vals = np.arange(0.01, 1, .01)
ce_vals = [binary_cross_entropy(p, q) for q in q_vals]

plt.figure()
plt.plot(q_vals, ce_vals)
plt.show()


