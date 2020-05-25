import sys
from matplotlib import pyplot as plt
import numpy as np

data = np.array([float(line.split()[-1]) for line in sys.stdin.readlines()])

r_avg = np.cumsum(data)/(np.arange(len(data))+1)
x = list(range(len(data)))

plt.plot(x, data, label = "acc")
plt.plot(x, r_avg, label = "r_avg")

plt.savefig(sys.argv[1], dpi=900)
