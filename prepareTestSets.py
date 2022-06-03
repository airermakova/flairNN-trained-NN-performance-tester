from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import re
import nltk
import numpy
import random
import codecs
from threading import Thread
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

trainSet = []
users = []
globalUsers = []
globalCount = []

#TO GET PHRASES
def getPhrasesFromFile(fileName):
    phrases= []    
    tokenized_phrases = []
    try:
       with codecs.open(fileName, 'r', "utf-8") as file:
           phrases = file.read().split(".")
       return phrases
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



#TO GET DATA FROM TEST FILE AND USERS FILE

trainSet = getPhrasesFromFile("testData.txt")
print("Taken phrases - ")
print(len(trainSet))
users = getDataFromFile("usersList.txt")
print("Taken users - ")
print(len(users))



def checkMarkedArrayPresence(phrases, users):
    onlyUsers = []
    for phrase in phrases:
        nltk_tags = pos_tag(word_tokenize(phrase))  
        iob_tagged = tree2conlltags(nltk_tags) 
        userFound = False
        for user in users:
            ind = phrase.find(user)
            us = user.split(" ")
            if ind>0:
               iob_tagged = markUser(iob_tagged, us)
        for iob in iob_tagged:
            if iob[2]=="B":
                onlyUsers.append(iob_tagged)          
    return onlyUsers
         

       
def markUser(phrase,user):
    finTuple = []
    users = []
    found = False
    print(user)
    exists = user in globalUsers
    if exists == False:
        globalUsers.append(user)
        globalCount.append(0)
    else:
        ind = globalUsers.index(user)
        globalCount[ind] = globalCount[ind]+1
    print("Length of all detected users " + str(len(globalCount)))
    
    if len(user)>1:
       nonFirst = False
       cnt = 0
       ucnt = 0
       for ph in phrase:
           finTuple.append(ph)
           ucnt = ucnt +1
           found = False
           for u in user: 
               if ph[0]==u:
                   found = True
           if found == True:
               cnt = cnt + 1
           else:
               cnt = cnt - 1
           if cnt == len(user)-1:              
               ls = list(phrase[ucnt-cnt-1])
               ls[2]="B"
               finTuple[ucnt-cnt-1]=tuple(ls)
               for i in range(cnt):
                   ls = list(phrase[ucnt-cnt+i])
                   ls[2]="I"
                   finTuple[ucnt-cnt+i]=tuple(ls)
               cnt = 0
    else:
        for i in range(0, len(phrase)):
           finTuple.append(phrase[i])         
           if phrase[i][0] == user[0] and phrase[i][1][0]=="N":  
               cnt = i-1
               found = False
               while cnt>=0 and phrase[cnt][1]=="JJ":
                   ls = list(phrase[cnt])
                   ls[2]="I"
                   finTuple[cnt] = (tuple(ls))
                   cnt = cnt - 1 
                   found = True

               ls = list(phrase[i])
               if found == False:
                   ls[2]="B"
               else:
                   ls[2]="I"
               finTuple[len(finTuple)-1] = (tuple(ls))
               if found == True:
                   ls = list(phrase[cnt+1])
                   ls[2]="B"
                   finTuple[cnt+1] = (tuple(ls))
    return finTuple
 
def writeResultFile(trainSet):
    t = open("train2.txt", "a")
    v = open("val2.txt", "a")
    te = open("test2.txt", "a")
    for arr in trainSet:
        for s in arr:
            for st in s:
                t.write((st + " ").encode("utf-8").decode("utf-8"))
            t.write("\n")            
        t.write("\n\n\n") 
    t = 0
    for arr in trainSet:
        t = t+1
        if t>3:
            t = 0
            for s in arr:
                for st in s:
                    v.write((st + " ").encode("utf-8").decode())
                v.write("\n")
            v.write("\n\n\n") 
    t = 0
    for arr in trainSet:
        t = t+1
        if t>4:
            t = 0
            for s in arr:
                for st in s:
                    te.write((st + " ").encode("utf-8").decode())
                te.write("\n")
            te.write("\n\n\n") 
     
         
def writeUsersFile():
    if len(globalUsers) != len(globalCount):
        return
    usersFile.write("\n\n\n\n")
    for i in range(0, len(globalCount)):
        for gl in range(0, len(globalUsers[i])):
            usersFile.write(str(globalUsers[i][gl]) + " ") 
        usersFile.write(": ")
        usersFile.write(str(globalCount[i]))
        usersFile.write("\n")  


#TO PREPARE USERS RECOGNITION FOR MULTITHREADING
def writeUsers(trainSet, users):
    fin = checkMarkedArrayPresence(trainSet, users)
    writeResultFile(fin)
    writeUsersFile()

#MAIN SCRIPTS

users = []
users = getDataFromFile("usersList.txt")
#print(users)


f = open("onlyUsers.txt", "w")
f.close()

rep = 0
i=0
phrases = []
usersFile = open("onlyDetectedUsersMulti.txt", "a")
for ph in trainSet:
    i=i+1
    phrases.append(ph)
    if i>=100:
        thread = Thread(target=writeUsers, args=(list(phrases), users))
        thread.start()
        i=0
        phrases = []
        rep = rep + 1







