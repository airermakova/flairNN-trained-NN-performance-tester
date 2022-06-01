from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
#import sys
import re
import nltk
import numpy
import random
import codecs
from langdetect import detect
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

#TO MARK USERS IN PHRASES 

def markUsers(phrases):
    finArr = []
    for phrase in phrases:
       finArr.append(getUsersFromNN(phrase)) 
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
    tupleArr = onlusers[1].split(",")
    for tp in tupleArr: 
        t = tp.split("/")
        if len(t)>1:
            if t[1] == "<unk>" or t[1] == "<unk>]":
                t[1] = "O"
            onlyUsers.append(tuple(t))
    return onlyUsers

#TO STORE MARKED PHRASES IN FINAL FILE

def writeUsersFile(finalArray):
    f = open("markedPhrases.txt", "w")
    for arr in finalArray:
        f.write(' '.join(str(s) for s in arr) + "\n\n")  


def writeOnlySimpleUsersFile(finalArray):
    f = open("markedSimpleUsersOnly.txt", "w")
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
            f.write(' '.join(str(s) for s in finalArray[i]) + "\n\n")  



def writeOnlyComUsersFile(finalArray):
    f = open("markedComplexUsersOnly.txt", "w")
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
            f.write(' '.join(str(s) for s in finalArray[i]) + "\n\n") 


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
    f = open("markedUsersStatistics.txt", "w")
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
    for i in range(0, len(allSimUsers)-1):
        tw = allSimUsers[i] + "-" + str(usersCounts[i]) + "\n"
        f.write(tw)
    for i in range(0, len(allComUsers)-1):
        tw = allComUsers[i] + "-" + str(comUsersCounts[i])+ "\n"
        f.write(tw)
            


#MAIN SCRIPT
#dataFileName = str(sys.argv[1])
#print(dataFileName)
#phNum = str(sys.argv[2])
#print(phNum)

phrases = getPhrasesFromFile("testData.txt", 0, 1000)
print("Taken phrases - ")
print(len(phrases))
users=markUsers(phrases)
#print(users)
writeUsersFile(users)
writeOnlySimpleUsersFile(users)
writeOnlyComUsersFile(users)
getUsersStatistics(users)
writeUsersStatFile()
print(simpleUsers)
print(complexUsers)