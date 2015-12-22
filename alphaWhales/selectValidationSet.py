import numpy as np
import os
import re
from collections import defaultdict
from scipy import random

validationDir = 'validation'
if not os.path.exists(validationDir):
    os.makedirs(validationDir)

output = np.genfromtxt('whales.csv', skip_header=1, dtype=[('image', 'S10'), ('label', 'S11')], delimiter=',')
whales = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in output]
images = [int(re.search("w_(\\d+)\.jpg", x[0]).group(1)) for x in output]

labelsDict = defaultdict(lambda:-1, {})

filenames = []

testset = []

indir = 'train'
for file in os.listdir(indir):
   if(file.endswith('.jpg')):
      filenames.append(file)

testdir = 'test'
for file in os.listdir(testdir):
   if(file.endswith('.jpg')):
      testset.append(file)

for i in xrange(len(whales)):
    if(labelsDict[whales[i]] == -1):
        labelsDict[whales[i]] = []
    labelsDict[whales[i]].append(images[i])

testset = [int(re.search("w_(\\d+)\.jpg", x).group(1)) for x in testset]

for w in set(whales):
    allExamplesForW = labelsDict[w]
    allExamplesForW = [x for x in allExamplesForW if x not in testset]
    allExamplesForW = random.permutation(allExamplesForW)
    for i in allExamplesForW[0:(len(allExamplesForW)/2)+(random.randint(0,(len(allExamplesForW))%2+1))]:
        print("copying %d\n"%i)
        os.rename(("%s/w_%d.jpg") % (indir, i), ("%s/w_%d.jpg") %(validationDir, i))


