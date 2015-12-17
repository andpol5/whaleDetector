#! /usr/bin/python
# Recreate the csv file from the available directories
import os

for root, dirs, files in os.walk(os.getcwd()):
    for directory in dirs:
        #import ipdb; ipdb.set_trace()
        for root2, dirs2, files2 in os.walk(os.path.join(root, directory)):
            for f in files2:
                print f + "," + directory


