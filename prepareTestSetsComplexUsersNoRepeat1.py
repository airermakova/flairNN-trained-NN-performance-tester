from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import PySimpleGUI as sg
import re
import os
import subprocess
import platform
import sys
import nltk
import numpy
import random
import shlex
import codecs
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

allFinished = False
trName = "trainnrC1.txt"
vName = "valnrC1.txt"
teName = "testnrC1.txt"
userStatisticsName = "onlyDetectedUsersNR1.txt"
tr = open(trName, "w")
v = open(vName, "w")
te = open(teName, "w")
usersFile = open(userStatisticsName, "w")
tr.close()
v.close()
te.close()
usersFile.close()


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
      window['-TEXT-'].update("Error in reading " + fileName)
      exit()


#TO MARK USERS AS WE HAVE DONE FOR TRAIN SET

def getDataFromFile(fileName):
    users = []        
    try:
       word_file = open (fileName, "r", encoding='utf-8')
       print("file object created")
       for l in word_file:
           user = l.replace('\r', '').replace('\n', '')
           #    user = stemmer.stem(m)
           #else:
           #    user = m
           users.append(user)
       return users
    except:
      print("Error in reading " + fileName)
      window['-TEXT-'].update("Error in reading " + fileName)
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
      window['-TEXT-'].update("Error in reading " + fileName)
      if len(users)>0:
          print(users[len(users)-1])
      exit()

def checkMarkedArrayPresence(phrases, users):
    onlyUsers = []
    registeredUser = []
    for phrase in phrases:
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
        allowAppend = False         
        for iob in iob_tagged:
             if iob[2]=="B" :
                 regStr = iob[0] 
                 newLen = newLen + 1                
             if iob[2]=="I":
                 regStr += " " + iob[0]
                 newLen = newLen + 1
             if iob[2]=="O" and len(regStr)>0:
                 registeredUser.append(regStr)
                 if len(regStr.split(" "))>1:
                     allowAppend = True
                 elif len(regStr)>0 and regStr[len(regStr) - 2:]=="er":
                     allowAppend = True
                 regStr = ""
                 newLen = newLen + 1
        if newLen>0:
            if len(regStr)>0 and regStr[len(regStr) - 2:]=="er" or len(regStr)>0 and " " in regStr:
                 allowAppend = True
            for us in registeredUser:
                if us not in foundUsers or foundUsers.count(us)<10:
                    foundUsers.append(us)
                    allowAppend = True
            if allowAppend == True:
                onlyUsers.append(iob_tagged)
                print(iob_tagged)
                window['-TEXT1-'].update(iob_tagged)
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
 
def writeResultFile(trainSet):
    with open(trName, 'a') as f:
        for arr in trainSet:
            for s in arr:
                for st in s:
                    try:
                        f.write((st + " ").encode("utf-8").decode())
                    except:
                        print("error")
                f.write("\n")            
            f.write("\n\n\n") 
        t = 0
    with open(vName, 'a') as v:
        for arr in trainSet:
            t = t+1
            if t>3:
                t = 0
                for s in arr:
                    for st in s:
                        try:
                            v.write((st + " ").encode("utf-8").decode())
                        except:
                            print("error")
                    v.write("\n")
                v.write("\n\n\n") 
        t = 0
    with open(teName, 'a') as te:
        for arr in trainSet:
            t = t+1
            if t>4:
                t = 0
                for s in arr:
                    for st in s:
                        try:
                            te.write((st + " ").encode("utf-8").decode())
                        except:
                            print("error")
                    te.write("\n")
                te.write("\n\n\n") 
     
         
def writeUsersFile():
    if len(globalUsers) != len(globalCount):
        return
    usersFile = open(userStatisticsName, "a")
    usersFile.write("\n\n\n\n")
    for i in range(0, len(globalCount)):
        for gl in range(0, len(globalUsers[i])):
            usersFile.write(str(globalUsers[i][gl]) + " ") 
        usersFile.write(": ")
        usersFile.write(str(globalCount[i]))
        usersFile.write("\n") 


#TO PREPARE USERS RECOGNITION FOR MULTITHREADING
def writeUsers():
    phr = []  
    threadsL.append(1) 
    while len(trainSet)>0:
        try: 
            sema.acquire()
            if len(trainSet)>=10:
                for i in range(0,10):
                    phr.append(trainSet[i]) 
            else:
                phr = trainSet
            for i in range(0,len(phr)):
                trainSet.pop(0)   
            sema.release()  
            fin = checkMarkedArrayPresence(phr, users)
            if len(fin)>0:
                writeResultFile(fin)
        except:
             print("Exception")
    threadsL.pop(0)
    print("THREAD FINISHED " + str(len(threadsL)))
    window['-TEXT1-'].update("THREAD FINISHED " + str(len(threadsL)))
    if len(threadsL)<=1:
        writeUsersFile()
        print("STATISTICS WRITTEN ")
        window['-TEXT1-'].update("TASK FINISHED. USER STATISTICS WRITTEN")
        window['-USSTAT-'].update("User file written " + userStatisticsName)
        window['Open folder'].update(visible=True)
        window['-TRAIN-'].update("User file written " + trName)
        window['-VAL-'].update("User file written " + vName)
        window['-TEST-'].update("User file written " + teName)
        allFinished = True
        

#MAIN SCRIPTS
def readPhrases():
    #dataFileName = str(sys.argv[1])
    print(dataFileName)
    #phNum = int(sys.argv[2])
    print(phNum)
    trainSet = getPhrasesFromFile(dataFileName, phStart, phNum)
    print('Taken phrases - ' + str(len(trainSet)))
    users = getDataFromFile("usersList.txt")
    print("Taken users - "+str(len(users)))
    window['-TEXT-'].update('Taken phrases - ' + str(len(trainSet)) + " Users taken from users list- "+str(len(users)))
    goldsets = getWordsFromGoldenSet("GoldenSet.txt")
    print("Taken goldsets - ")
    print(len(goldsets))
    usersFile = open("onlyDetectedUsersNR1.txt", "a")
    return trainSet

def readUsers():
    users = getDataFromFile("usersList.txt")    
    #print(users)
    return users

def startThreads():
    window['Open folder'].update(visible=False)
    rep = 0
    i=0
    #writeUsers(trainSet, users)
    threads = []
    for cnt in range(0,10):
        threads.append(Thread(target=writeUsers))
    for ph in threads:
        print("thread start")
        ph.start()

#OPEN COMMAND

def IButton(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0))

#USER INTERFACE

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
parameters_list_column = [
    [sg.Text(' ')], 
            [sg.Text('Entrance parameters')],
            [sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse()],
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
    [sg.Text('Results:', key='-USSTAT-', background_color='#DAE0E6', text_color='black')],
    [sg.Text('', key='-TRAIN-', background_color='#DAE0E6', text_color='black')],
    [sg.Text('', key='-TEST-', background_color='#DAE0E6', text_color='black')],
    [sg.Text('', key='-VAL-', background_color='#DAE0E6', text_color='black'), IButton('Open folder',visible=False)],
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
    if event == 'Open folder': 
        subprocess.Popen(f'explorer {os.path.abspath(os.getcwd())}')
    if event == 'Ok': 
        print('You entered ', values[0] + values[1])
        dataFileName = str(values[0])
        try:
            phStart = int(values[1])
        except:
            phStart = 0
        try:
            phNum = int(values[2])
        except:
            phNum = 0
        trainSet = readPhrases()
        users = readUsers()
        startThreads()
    
window.close()










