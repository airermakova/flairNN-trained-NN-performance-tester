import sys
import re
import nltk
import numpy
import random
import codecs

fileName = str(sys.argv[1])
print("Source File - " + fileName)

resFileName = str(sys.argv[2])
print("Destination File - " + resFileName)
resF = open(resFileName, "w")

try:
    word_file = open (fileName, "r", encoding='latin-1')
    print("file object created")
    for line in word_file:
        try:
            line1 = bytes(line, 'latin-1').decode('utf-8', 'ignore')
            resF.write(line1)
        except: 
            print("Failed phrase - "+line)
        #print(line1)
except:
    print("Error in reading " + fileName)
    exit()

