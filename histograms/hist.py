import numpy as np
import re
import matplotlib.pyplot as plt

file1 = '../alphaWhales/train/validation/train.csv'
file2 = '../alphaWhales/train/train/train.csv'
fileTrain = '../alphaWhales/train/train/train.csv'
data1 = np.genfromtxt(file1, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')
data2 = np.genfromtxt(file2, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')
dataTrain = np.genfromtxt(fileTrain, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')
labels1 = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in data1]
labels2 = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in data2]
labelsTrain = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in dataTrain]
labels = labels1 + labels2

counts = np.bincount(labels)
ii = np.unique(labelsTrain)


plt.bar(range(len(ii)), counts[ii])
plt.title("")
plt.xlabel("Whale")
plt.ylabel("Frequency")
plt.xticks(range(len(ii)), ii)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)

plt.show()




