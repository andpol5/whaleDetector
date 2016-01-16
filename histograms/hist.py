import numpy as np
import re
import matplotlib.pyplot as plt

file = '../alphaWhales/train/validation/train.csv'
fileTrain = '../alphaWhales/train/train/train.csv'
data = np.genfromtxt(file, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')
dataTrain = np.genfromtxt(fileTrain, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')
labels = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in data]
labelsTrain = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in dataTrain]
counts = np.bincount(labels)
ii = np.unique(labelsTrain)



plt.bar(range(len(ii)), counts[ii])
plt.title("Test Set Histogram")
plt.xlabel("Whale")
plt.ylabel("Frequency")
plt.xticks(range(len(ii)), ii)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)

plt.show()




