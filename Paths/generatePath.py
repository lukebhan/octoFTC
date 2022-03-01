import numpy as np

count = 0
totalVal  =0
f = open('xlinetraj.csv', 'w')
f2 = open('ylinetraj.csv', 'w')
while totalVal < 10:
    f.write(str(count) + "," + str(0) + "\n")
    f2.write(str(count) + "," + str(totalVal) + "\n")
    totalVal += 0.01 
    count  += 0.01



