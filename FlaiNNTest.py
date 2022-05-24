from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import re
import sys
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
users = []

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

#TO GET USERS FROM LIST OF USERS
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


#TO MARK USERS AS WE HAVE DONE FOR TRAIN SET

def checkUsersPresence(phrase, users):
    checkUsers = False
    pr = []
    wr = []
    finArr = []
    onlyUsers = []
    phrasesArr=[]
    cancelId = True
    nltk_tags = pos_tag(word_tokenize(phrase))  
    iob_tagged = tree2conlltags(nltk_tags) 
    userFound = False
    for user in users:
        ind = phrase.find(user)
        us = user.split(" ")
        if ind>0:
           iob_tagged = markUser(iob_tagged, us)
    onlyUsers.append(iob_tagged)          
    return onlyUsers
                
def markUser(phrase,user):
    finTuple = []
    users = []
    found = False
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

def prepareMarkedUsers(trainPrep):
    trainUs = []
    for tr in trainPrep:
        for t in tr:
            trainUs.append(t)
    return trainUs


#TO GET USERS FROM CLASSIFIER

model = SequenceTagger.load("C:/Users/airer/Documents/Pisa/Classifier/trainer/final-model.pt")

def getUsersFromNN(phrase):
    # create example sentence
    sentence = Sentence(phrase)
    # predict tags and print
    model.predict(sentence)
    sent = sentence.to_tagged_string()
    onlusers = sent.split("[")
    #create tuples
    if len(onlusers)<2:
        t=[""]
        tuple(t)
        return t
    tupleArr = onlusers[1].split(",")
    print(onlusers[1])
    users=[]
    for tp in tupleArr: 
        t = tp.split("/")
        if len(t)>1:
            if t[1] == "<unk>" or t[1] == "<unk>]":
                t[1] = "O"
            users.append(tuple(t))
    return users

#TO CALCULATE TRUE AND FALSE POSITIVES

def precisionEstimator(clasRes, trainPrep):
    results = [0,0,0,0]
    if clasRes == None:
        return results
    diff = len(clasRes)
    if len(clasRes) < len(trainPrep):
        diff = len(clasRes)
    elif len(clasRes) > len(trainPrep):
        diff = len(trainPrep)
    for i in range(diff):
        c = re.sub(r'\W+', '', clasRes[i][0])
        u = re.sub(r'\W+', '', trainPrep[i][0])
        if u==c:
             us = re.sub(r'\W+', '', trainPrep[i][2])
             cl = re.sub(r'\W+', '', clasRes[i][1])
             if us.lower() == "b" or us.lower() == "i":
                 results[0] = results[0]+ 1
             if us.lower() == cl.lower():
                 results[1]= results[1]+ 1  
             elif cl.lower() == "b" or cl.lower() == "i" and us.lower()=="o": 
                 results[2]= results[2]+ 1 
             elif us.lower() == "b" or us.lower() == "i" and cl.lower()=="o": 
                 results[3]= results[3]+ 1 
    return results        

def equalizeArrays(clasRes, trainPrep):
    if len(clasRes) < len(trainPrep):
        diff = len(trainPrep)-len(clasRes)
        for i in range(len(clasRes), diff-1):
            ls = list(trainPrep[i])
            ls[1] = "O"
            clasRes.append(tuple(ls))
        return clasRes
    if len(clasRes) > len(trainPrep):
        diff = len(clasRes) - len(trainPrep)
        for i in range(len(trainPrep), diff-1):
            ls = list(clasRes[i])
            ls[1] = "O"
            trainPrep.append(tuple(ls))
        return trainPrep


#MAIN VARIABLES
realPositives = 0
truePositives = 0
falsePositives = 0
falseNegatives = 0

#MAIN SCRIPT
users = getDataFromFile("usersList.txt")
fileName = str(sys.argv[1])
print(fileName)
phrases = getPhrasesFromFile(fileName)
print(phrases)

for phrase in phrases:
    print(phrase)
    trainPrep = checkUsersPresence(phrase, users) 
    if len(trainPrep)>0:
        print(trainPrep[0])
        trainUs = prepareMarkedUsers(trainPrep)     
        clasRes = getUsersFromNN(phrase)
        if len(clasRes)>0:
             if len(clasRes) > len(trainPrep):
                 trainPrep = equalizeArrays(clasRes, trainPrep)
             else:
                 clasRes = equalizeArrays(clasRes, trainPrep)
             res = precisionEstimator(clasRes,trainUs)
             realPositives = realPositives + res[0]
             truePositives = truePositives + res[1]
             falsePositives = falsePositives + res[2]

precision = truePositives/(truePositives+falsePositives)
recall = truePositives/(truePositives+falseNegatives)
f1 = 2*(precision*recall/(precision +recall))


print("precision - " + str(precision))
print("recall - " + str(recall))
print("f1 score - " + str(f1))


