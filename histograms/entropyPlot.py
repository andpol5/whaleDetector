import matplotlib.pyplot as plt
import numpy as np
import sys

prefix = '../alphaWhales/'
files = sys.argv[1:]
entropies = []
for file in files:
   f = open(prefix + file, 'r')
   for line in f:
      if (line.startswith("train cross entropy: ")):
         entropy = float(line.split('train cross entropy: ')[1][:-1])
         entropies.append(entropy)
# logs = np.zeros(len(entropies))
logs = np.log10(np.array(entropies))
plt.plot(logs)
x,p = plt.xticks()
xticks = [str(t*25) for t in x]
plt.xticks(x, xticks)

y,p = plt.yticks()
yticks = [str(round(pow(10,t))) for t in y]
plt.yticks(y, yticks)

plt.show()
