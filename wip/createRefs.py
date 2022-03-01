import numpy as np
import pandas as pd

path = "../Paths/"

filename = "xrefCirctraj.csv"

filename2 = "yrefCirctraj.csv"


refarr = pd.read_csv(path+filename, header=None).iloc[:, 1]

refarr1 = pd.read_csv(path+filename2, header=None).iloc[:, 1]
np.savetxt('xref2', refarr)
np.savetxt('yref2', refarr1)
