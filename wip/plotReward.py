import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
arr = []
count = 0
arr2= []
with open('tmp/monitor.csv') as f:
    line = f.readline()
    while line:
        if(count < 2):
            count += 1
            line = f.readline()
        else:
            val = line.split(",")
            line=f.readline()
            arr.append(float(val[0]))
            arr2.append(np.mean(arr[-15:]))


plt.plot(np.array(arr))
plt.plot(np.array(arr2), label="Last 5")
plt.legend()
plt.savefig("fig.png")
plt.show()
