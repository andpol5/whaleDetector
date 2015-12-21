import numpy as np
import os
import re
from collections import defaultdict
from scipy import random

testDir = 'test'

if not os.path.exists(testDir):
    os.makedirs(testDir)

output = np.genfromtxt('whales.csv', skip_header=1, dtype=[('image', 'S10'), ('label', 'S11')], delimiter=',')
whales = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in output]
images = [int(re.search("w_(\\d+)\.jpg", x[0]).group(1)) for x in output]

labelsDict = defaultdict(lambda:-1, {})

filenames = []

indir = 'alphaWhales-001/imgs'
for file in os.listdir(indir):
   if(file.endswith('.jpg')):
      filenames.append(file)

for i in xrange(len(whales)):
    if(labelsDict[whales[i]] == -1):
        labelsDict[whales[i]] = []
    labelsDict[whales[i]].append(images[i])




for w in set(whales):
    allExamplesForW = labelsDict[w]
    allExamplesForW = random.permutation(allExamplesForW)
    for i in allExamplesForW[0:5]:
        print("copying %d\n"%i)
        os.rename((indir + "/w_%d.jpg") % i, (testDir + '/w_%d.jpg') %i)


