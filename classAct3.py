import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-10, 10, 0.001)

y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.show()

q = np.arange(0.001, 1, 0.001)
p = .8
l = -p * np.log(q) - (1 - p) * np.log(1 - q)

plt.figure()
plt.plot(q, l)
plt.show()
