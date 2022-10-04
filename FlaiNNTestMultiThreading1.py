from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import re
import os
import sys
import nltk
import numpy
import random
import subprocess
import codecs
import webbrowser
import PySimpleGUI as sg
from threading import *
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
goldsets = []
globalUsers = []
globalCount = []
foundUsers = []
threadsL = []
sema = Semaphore(1)
stemmer = SnowballStemmer("english")
truePositives = []
truePositivesUs = []
falsePositives = []
falsePositivesUs = []
falseNegatives = []
falseNegativesUs = []
totalNumber = []


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


#TO GET ROOTS FROM GOLDEN SETS
def getWordsFromGoldenSet(fileName):
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

def checkMarkedArrayPresence(phrase, users):
    onlyUsers = []
    registeredUser = []
    nltk_tags = pos_tag(word_tokenize(phrase))  
    iob_tagged = tree2conlltags(nltk_tags) 
    userFound = False
    for user in users:
        ind = phrase.find(user)
        us = user.split(" ")
        if ind>0:
           iob_tagged = markUser(iob_tagged, us)
    regStr = ""
    newLen = 0       
    for iob in iob_tagged:
         if iob[2]=="B" :
             regStr = iob[0]                 
         if iob[2]=="I":
             regStr += " " + iob[0]
         if iob[2]=="O" and len(regStr)>0:
             registeredUser.append(regStr)
             regStr = ""
             newLen = newLen + 1
    if newLen>0:
       onlyUsers.append(iob_tagged)
       #print(iob_tagged)
    return onlyUsers         

       
def markUser(phrase,user):
    finTuple = []
    users = []
    found = False
    #print(user)
    exists = user in globalUsers
    if exists == False:
        globalUsers.append(user)
        globalCount.append(0)
    else:
        ind = globalUsers.index(user)
        globalCount[ind] = globalCount[ind]+1
    #print("Length of all detected users " + str(len(globalCount)))
    
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
               while cnt>=0 and phrase[cnt][1]=="JJ" or cnt>=0 and phrase[cnt][1]=="NNP" or cnt>=0 and phrase[cnt][1]=="VBN":
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



#TO GET USERS FROM CLASSIFIER

model = SequenceTagger.load("C:/Users/airer/Documents/Pisa/Classifier/trainerNRFinal10Rep/final-model.pt")

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
    #print(onlusers[1])
    users=[]
    for tp in tupleArr: 
        t = tp.split("/")
        if len(t)>1:
            if t[1] == "<unk>" or t[1] == "<unk>]":
                t[1] = "O"
            users.append(tuple(t))
    return users

#TO WRITE RESULTS

def writeResults():
    tr = open("NNStatistics.txt", "w")
    tr.write(" total users number - ")
    tr.write(str(totalNumber[0]))
    tr.write("\n")
    tr.write(" true positives number - ")
    tr.write(str(truePositives[0]))
    tr.write("\n")
    tr.write(" false positives number - ")
    tr.write(str(falsePositives[0]))
    tr.write("\n")
    tr.write(" false negatives number - ")
    tr.write(str(falseNegatives[0]))
    tr.write("\n")
    prec = truePositives[0]/(truePositives[0]+falsePositives[0])
    rec = truePositives[0]/(truePositives[0]+falseNegatives[0])
    fscore = 2*(prec*rec/(prec+rec))
    tr.write(" precision - "+str(prec))
    tr.write("\n")
    tr.write(" recall - " + str(rec))
    tr.write("\n")
    tr.write(" f score - " + str(fscore))
    tr.write("\n")
    tr.write("\n")
    tr.write("false positives:\n")
    for us in falsePositivesUs:
        tr.write(us + "\n")
    tr.write("\n")
    tr.write("\n")
    tr.write("false negatives:\n")
    for us in falseNegativesUs:
        tr.write(us + "\n")
    tr.write("\n")
    tr.write("\n")
    tr.write("true positives:\n")
    for us in truePositivesUs:
        tr.write(us + "\n")
    
    window['-USSTAT-'].update("Total number of detected users " + str(totalNumber[0]) + " Total number of true positives " + str(truePositives[0]) + " Total number of false positives " + str(falsePositives[0]) + " Total number of false negatives " + str(falseNegatives[0]) + " written in file NNStatistics.txt" + " precision - "+str(prec) + " recall - " + str(rec) + " f score - " + str(fscore))



#TO PREPARE USERS RECOGNITION FOR MULTITHREADING

def writeUsers():
    phr = ""    
    threadsL.append(1) 
    while len(trainSet)>0:
        try: 
            sema.acquire()
            if len(trainSet)>0:
                phr = trainSet[0]
                trainSet.pop(0)  
            sema.release()  
            window['-USSTAT-'].update("Remaining phrases "+ str(len(trainSet)))
            fin = checkMarkedArrayPresence(phr, users)
            mrUsersNN = getUsersFromNN(phr)
            if len(fin)>0:      
                mrUsers = fin[0]
                if len(mrUsers)>0 and len(mrUsers) == len(mrUsersNN):
                   cnt = len(mrUsersNN)
                   for i in range(0, cnt):
                       tupNN = tuple(mrUsersNN[i])
                       tup = tuple(mrUsers[i])
                       strC = re.sub(r'\W+', '', tup[0])
                       strNN = re.sub(r'\W+', '', tupNN[0])
                       usC = re.sub(r'\W+', '', tup[2])
                       usNN = re.sub(r'\W+', '', tupNN[1]) 
                       window['-TEXT1-'].update("Taken users: "  + strC + " " + strNN)                
                       if strC == strNN:
                           if usC == "B" or usC == "I":
                               totalNumber[0] = totalNumber[0] + 1
                               print(strC)
                               if usC == usNN:
                                   truePositives[0] = truePositives[0] + 1
                                   truePositivesUs.append(strC)
                           if usC != usNN and usC == "O":
                               falsePositives[0] = falseNegatives[0] + 1   
                               falsePositivesUs.append(str(tupNN[0]))            
                           if usC != usNN and usNN == "O":
                               falseNegatives[0] = falsePositives[0] + 1
                               falseNegativesUs.append(str(tupNN[0]))
        except:
             print("Exception")
    threadsL.pop(0)
    print("THREAD FINISHED " + str(len(threadsL)))
    if len(threadsL)<=1:
        print("Total number of detected users " + str(totalNumber[0]))
        print("Total number of true positives " + str(truePositives[0]))
        print("Total number of false positives " + str(falsePositives[0]))
        print("Total number of false negatives " + str(falseNegatives[0]))        
        window['Open'].update(visible=True)
        writeResults()
        print("STATISTICS WRITTEN ")

#TO CALCULATE TRUE AND FALSE POSITIVES


#MAIN SCRIPTS


#TO GET DATA FROM TEST FILE AND USERS FILE

def setUpTrainSet(dataFileName, st, phNum):
    window['Open'].update(visible=False)
    window['-USSTAT-'].update("Results")
    trainSet = getPhrasesFromFile(dataFileName, st, phNum)
    print("Taken phrases - "+str(len(trainSet)))
    #goldsets = getWordsFromGoldenSet("GoldenSet.txt")
    #print("Taken goldsets - ")
    #print(len(goldsets))
    totalNumber.append(0)
    truePositives.append(0)
    falsePositives.append(0)
    falseNegatives.append(0)
    return trainSet

def setUpUsers():
    users = getDataFromFile("usersList.txt")
    print("Taken users - "+str(len(users)))
    #print(users)
    totalNumber.append(0)
    truePositives.append(0)
    falsePositives.append(0)
    falseNegatives.append(0)
    return users

def startThreads():
   threads = []
   for cnt in range(0,10):
       threads.append(Thread(target=writeUsers))

   for ph in threads:
        print("thread start")
        ph.start()
   return threads


#USER INTERFACE

def IButton(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0), background_color='#DAE0E6')

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
parameters_list_column = [
            [sg.Text(' ')], 
            [sg.Text('Entrance parameters')],
            [sg.Text("Choose trained model: "), sg.Input(), sg.FileBrowse()],
            [sg.Text("Choose text file: "), sg.Input(), sg.FileBrowse()],
            [sg.Text('Enter initial phrase you wish to process from (to start from beginning leave blank)'), sg.InputText()],
            [sg.Text('Enter number of phrases you wish to process (to process all phrases leave blank)'), sg.InputText()],
            [sg.Text(' ')]
]
button_list_column = [
    [sg.Button('Ok', button_color=(sg.YELLOWS[0], sg.GREENS[0])), sg.Button('Exit',button_color=(sg.YELLOWS[0], sg.BLUES[0]))]
]
final_list_column = [
    [sg.Text('Input data:', key='-TEXT-',background_color='#DAE0E6', text_color='black')],
    [sg.MLine('Execution log', key='-TEXT1-', background_color='#DAE0E6', size=(90, 3), text_color='black')],
    [sg.MLine('Results', key='-USSTAT-', background_color='#DAE0E6', size=(50, 4), text_color='black'), IButton('Open',visible=False)]    
]
layout = [
            [sg.Column(parameters_list_column, justification='left')],
            [sg.Column(button_list_column, justification='right')],
            [sg.Column(final_list_column, justification='left', background_color='#DAE0E6', size=(700, 450))]
]

# Create the Window
window = sg.Window('Automatic preparation of neural network training set', layout, resizable=False, size=(700, 450))

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break
    if event == 'Open': 
        #subprocess.Popen(f'explorer {os.path.abspath(os.getcwd())}')
        webbrowser.open("NNStatistics.txt")
    if event == 'Ok': 
        trainSet.clear()
        sema.release()
        model = SequenceTagger.load(values[0])
        print('You entered ', values[1] + values[2])
        dataFileName = str(values[1])
        try:
            phStart = int(values[2])
        except:
            phStart = 0
        try:
            phNum = int(values[3])
        except:
            phNum = 0
        trainSet = setUpTrainSet(dataFileName, phStart, phNum)
        users = setUpUsers()

        window['-TEXT-'].update('Taken phrases - ' + str(len(trainSet)) + " Users taken from users list- "+str(len(users)))
        startThreads()
    
window.close()



