import os
import numpy as np

dir = 'batch2'
labelsFile = 'whales.csv'

filenames = []
for file in os.listdir(dir):
   if(file.endswith('.jpg')):
      filenames.append(file)

output = np.genfromtxt('whales.csv', skip_header=1, dtype=[('image', 'S10'), ('label', 'S11')], delimiter=',')

whales = [x[1] for x in output]
images = [x[0] for x in output]

f1 = open(dir + '/validation.csv', 'w+')


for file in filenames:
   ind = images.index(file)

   f1.write("\n%s,%s" % (file,whales[ind]))


f1.flush()
f1.close()
