import sys
from matplotlib import pyplot as plt
import numpy as np

data = np.array([float(line.split()[-1]) for line in sys.stdin.readlines()])

r_avg = np.cumsum(data)/(np.arange(len(data))+1)
x = list(range(len(data)))

plt.plot(x, data, label = "acc", color = "b")
plt.plot(x, r_avg, label = "r_avg", color = "r", linestyle="dashed")

plt.legend()

plt.title("Validation accuracy over time")
plt.xlabel("Time-Step")
plt.ylabel("Running Accuracy")

plt.savefig(sys.argv[1], dpi=900)
