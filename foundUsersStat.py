import re
import sys
import nltk
import numpy
import random
import codecs
from threading import Thread
from langdetect import detect


foundUsers = []


#TO GET PHRASES
def getPhrasesFromFile(fileName):
    phrases= []    
    tokenized_phrases = []
    try:
       with codecs.open(fileName, 'r', "utf-8") as file:
           phrases = file.read().split("\n")
       return phrases
    except:
      print("Error in reading " + fileName)
      exit()

foundUsers = getPhrasesFromFile("onlyDetectedUsersNR.txt")

usersFileC = open("onlyDetectedComplexUsersNRStat.txt", "w")
usersFileS = open("onlyDetectedSimpleUsersNRStat.txt", "w")
comUsers = []
simUsers = []
usLen = len(foundUsers)
print(usLen)
for i in range(0,5000):
    fu = foundUsers[usLen-i-1]
    if fu == None or fu == "":
        continue
    fum = fu.replace(" ",":").replace(":::",":").replace("::",":")
    print(fum)
    if len(fum.split(":"))>=3:
        if fu not in comUsers:
           usersFileC.write(fu)
           comUsers.append(fu)
    else:
        if fu not in simUsers:
           usersFileS.write(fu)
           simUsers.append(fu)

usersFileC.write("FOUND USERS TOTAL COUNT:")
usersFileC.write(str(len(comUsers)))


usersFileS.write("FOUND USERS TOTAL COUNT:")
usersFileS.write(str(len(simUsers)))
