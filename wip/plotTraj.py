import numpy as np
import matplotlib
import matplotlib.pyplot as plt

xref1 = np.loadtxt("xref1")
yref1 = np.loadtxt("yref1")

xref2 = np.loadtxt("xref2")
yref2 = np.loadtxt("yref2")

xref3 = np.loadtxt("xref3")
yref3 = np.loadtxt("yref3")

def load2darr(filename):
    f = open(filename, 'r')
    line = f.readline()
    split = line.split()
    f.close()
    # do this to reread
    resarr = []
    for i in range(len(split)):
        f = open(filename, 'r')
        line = f.readline()
        tmparr = []
        while line:
            s = line.split()
            tmparr.append(float(s[i]))
            line = f.readline()
        resarr.append(tmparr)
        f.close()
    f.close()
    return resarr

x_runs = load2darr("t1x")
y_runs = load2darr("t1y")
errors = np.loadtxt("t1error")
plt.plot(xref1, yref1, label="Ref")

for i in range(len(errors)):
    plt.plot(x_runs[i], y_runs[i], label="(Error: " + "{:.1f}".format(errors[i]) + ")")
    #plt.plot(rpms[i], label="{:.3f}".format(fault_params[i]) + "(Error: " + "{:.1f}".format(errors[i]) + ")")
plt.legend()
plt.show()

plt.figure()
x_runs = load2darr("t2x")
y_runs = load2darr("t2y")
errors = np.loadtxt("t2error")
plt.plot(xref2, yref2, label="Ref")

for i in range(len(errors)):
    plt.plot(x_runs[i], y_runs[i], label="(Error: " + "{:.1f}".format(errors[i]) + ")")
    #plt.plot(rpms[i], label="{:.3f}".format(fault_params[i]) + "(Error: " + "{:.1f}".format(errors[i]) + ")")
plt.legend()
plt.show()

plt.figure()
x_runs = load2darr("t3x")
y_runs = load2darr("t3y")
errors = np.loadtxt("t3error")
plt.plot(xref3, yref3, label="Ref")

for i in range(len(errors)):
    plt.plot(x_runs[i], y_runs[i], label="(Error: " + "{:.1f}".format(errors[i]) + ")")
    #plt.plot(rpms[i], label="{:.3f}".format(fault_params[i]) + "(Error: " + "{:.1f}".format(errors[i]) + ")")
plt.legend()
plt.show()


