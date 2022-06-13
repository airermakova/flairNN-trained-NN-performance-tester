from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import sys
import re
import nltk
import numpy
import random
import codecs
from langdetect import detect
from threading import Thread
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('conll2002')
nltk.download('conll2000')
nltk.download('brown')
nltk.download('universal_tagset')
from nltk import word_tokenize,pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words
from nltk.corpus import conll2000, conll2002

phrases = []
fphrases = []
users = []
simpleUsers = []
complexUsers = []
usersCounts = []
comUsersCounts = []
simpleF = open("markedSimpleUsersOnly1.txt", "a")
markedPhrasesFile = open("markedPhrases1.txt", "a")
complexPhrasesFile = open("markedComplexUsersOnly1.txt", "a")
userStatistics = open("markedUsersStatistics1.txt", "a")

model = SequenceTagger.load("C:/Users/airer/Documents/Pisa/Classifier/trainer/final-model.pt")

#TO GET PHRASES
def getPhrasesFromFile(fileName, st, fin):
    phrases= []  
    finArr = []  
    tokenized_phrases = []
    try:
       with codecs.open(fileName, 'r', "utf-8") as file:
           phrases = file.read().split(".")
       if fin == 0 or fin>len(phrases):
           fin = len(phrases)-1
       for i in range(st,fin):
           finArr.append(phrases[i])
       return finArr
    except:
      print("Error in reading " + fileName)
      exit()

#TO MARK USERS AS WE HAVE DONE FOR TRAIN SET

def getDataFromFile(fileName):
    users = []        
    try:
       word_file = open (fileName, "r", encoding='utf-8')
       print("file object created")
       for l in word_file:
           users.append(l.replace('\r', '').replace('\n', ''))
       return users
    except:
      print("Error in reading " + fileName)
      if len(users)>0:
          print(users[len(users)-1])
      exit()


dataFileName = str(sys.argv[1])
#print(dataFileName)
phNum = int(sys.argv[2])
print(phNum)

phrases = getPhrasesFromFile(dataFileName, 0, phNum)
print("Taken phrases - ")
print(len(phrases))


#TO MARK USERS IN PHRASES 

def markUsers(phrases):
    finArr = []
    for phrase in phrases:
       marked = getUsersFromNN(phrase)
       #print(marked)
       finArr.append(marked)
    return finArr    

   
def getUsersFromNN(phrase):
    # create example sentence
    sentence = Sentence(phrase)
    # predict tags and print
    model.predict(sentence)
    sent = sentence.to_tagged_string()
    onlusers = sent.split("[")
    #create tuples
    onlyUsers = []
    if len(onlusers)<2:
        return onlyUsers
    tupleArr = onlusers[1].split(",")
    if len(tupleArr)<1:
        return onlyUsers
    prev = tupleArr[0]
    for i in range(1, len(tupleArr)-1): 
        t = tupleArr[i].split("/")
        tp = tupleArr[i-1].split("/")
        if len(t)>1 and len(tp)>1:
            change = False
            if t[1] == "<unk>" or t[1] == "<unk>]":
                t[1] = "O"
            if tp[1] == "<unk>" or tp[1] == "<unk>]":
                tp[1] = "O"   
                     
            if t[1] == "I" and tp[1] == "B": 
                pos = list(pos_tag(phrase.split(" ")))
                for st in pos:
                    if st[0] in str(tp[0]) and "JJ" not in st[1] and "RB" not in st[1] and "DT" not in st[1] and "NN" not in st[1]:
                       change = True
                       print(st[1] + " " + tp[0])
                
            if change == True:
                t[1] = "B"
                tp[1] = "O"
                print(t)
                print(tp)
            if len(onlyUsers)>0:
                 onlyUsers[len(onlyUsers)-1] = tuple(tp)
            onlyUsers.append(tuple(t))
    return onlyUsers

#TO STORE MARKED PHRASES IN FINAL FILE

def writeUsersFile(finalArray):    
    for arr in finalArray:
        markedPhrasesFile.write(' '.join(str(s) for s in arr) + "\n\n")  


def writeOnlySimpleUsersFile(finalArray):
    for i in range(0, len(finalArray)-2):
        found = False
        wr = False
        arr = list(finalArray[i])
        for y in range(0, len(arr)-2):
            for s in arr[y]:
                 if s == "B":
                     found = True
            if found == True:
                for s in arr[y+1]:
                    if s!= "I":
                        wr = True
        if wr == True:
            simpleF.write(' '.join(str(s) for s in finalArray[i]) + "\n\n")  



def writeOnlyComUsersFile(finalArray):    
    for i in range(0, len(finalArray)-2):
        found = False
        wr = False
        arr = list(finalArray[i])
        for y in range(0, len(arr)-2):
            for s in arr[y]:
                 if s == "B":
                     found = True
            if found == True:
                for s in arr[y+1]:
                    if s == "I":
                        wr = True
        if wr == True:
            complexPhrasesFile.write(' '.join(str(s) for s in finalArray[i]) + "\n\n") 


def getUsersStatistics(finalArray):
    for i in range(0, len(finalArray)-2):
        found = False
        wr = False
        arr = list(finalArray[i])
        for y in range(0, len(arr)-2):
            for s in arr[y]:
                 suser = s
                 if s == "B":
                     found = True
            if found == True:
                for s in arr[y+1]:
                    if s!= "I":
                        wr = True
        if wr == True:
            for y in range(0, len(arr)-2):
                us = list(arr[y])
                if us[1] == "B":
                    simpleUsers.append(us[0]) 
    for i in range(0, len(finalArray)-2):
        found = False
        wr = False
        arr = list(finalArray[i])
        for y in range(0, len(arr)-2):
            for s in arr[y]:
                 suser = s
                 if s == "B":
                     found = True
            if found == True:
                for s in arr[y+1]:
                    if s == "I":
                        wr = True
        if wr == True:
            str = ""
            for y in range(0, len(arr)-2):
                us = list(arr[y])
                if us[1] == "B": 
                    str = us[0]
                if us[1] == "I":
                    str = str + " " + us[0]
                if us[1]!="B" and us[1]!="I" and len(str)>0:
                    complexUsers.append(str)
                    

def writeUsersStatFile():    
    allSimUsers = []
    allComUsers = []
    for user in simpleUsers:
         if user not in allSimUsers: 
             allSimUsers.append(user)
             usersCounts.append(1)
         else: 
             ind = allSimUsers.index(user)
             if ind>=0:
                 usersCounts[ind] = usersCounts[ind] + 1
    for user in complexUsers:
         if user not in allComUsers: 
             allComUsers.append(user)
             comUsersCounts.append(1)
         else: 
             ind = allComUsers.index(user)
             if ind>=0:
                 comUsersCounts[ind] = comUsersCounts[ind] + 1
    userStatistics.write("\n\n\n\n\n\n")
    for i in range(0, len(allSimUsers)-1):
        tw = allSimUsers[i] + "-" + str(usersCounts[i]) + "\n"
        userStatistics.write(tw)
    for i in range(0, len(allComUsers)-1):
        tw = allComUsers[i] + "-" + str(comUsersCounts[i])+ "\n"
        userStatistics.write(tw)
            


#MAIN SCRIPT

def writeUsers(phr, us):
    users=markUsers(phr)
    writeUsersFile(users)
    writeOnlySimpleUsersFile(users)
    writeOnlyComUsersFile(users)
    getUsersStatistics(users)
    writeUsersStatFile()
    print(simpleUsers)
    print(complexUsers)
    print("THREAD FINISHED " + str(us))

    

threads = []
i = 0
phr = []
cnt = 0
for ph in phrases:   
    i=i+1    
    phr.append(ph)
    if i>=10 and len(phr)<=10:
        arr = []
        arr = list(phr)
        threads.append(Thread(target=writeUsers, args=(arr, cnt)))
        cnt = cnt + 1
        i=0
        phr.clear()


for ph in threads:
    print("thread start")
    ph.start()
