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
import PySimpleGUI as sg
import os
import subprocess
import platform
import shlex
import codecs
from threading import *
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

threads = []
threadsL = []
phrases = []
fphrases = []
users = []
simpleUsers = []
complexUsers = []
usersCounts = []
comUsersCounts = []
allSimUsers = []
allComUsers = []
usersCounts = []
comUsersCounts = []
simpleF = open("markedSimpleUsersOnlyFullNN1.txt", "w")
markedPhrasesFile = open("markedPhrasesFullNN1.txt", "w")
complexPhrasesFile = open("markedComplexUsersOnlyFullNN1.txt", "w")
userStatistics = open("markedUsersStatisticsFullNN1.txt", "w")
usersNotInListFile = open("markedUsersOutOfList1.txt", "w")
simpleF = open("markedSimpleUsersOnlyFullNN1.txt", "w")
markedPhrasesFile.close()
complexPhrasesFile.close()
userStatistics.close()
usersNotInListFile.close()
simpleF.close()

markedPhrasesFileName = "markedPhrasesFullNN.txt"
sema = Semaphore(1)

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
      window['-TEXT1-'].update("Error in reading " + fileName)
      print("Error in reading " + fileName)
      exit()

#TO MARK USERS AS WE HAVE DONE FOR TRAIN SET

def getDataFromFile(fileName):
    users = []        
    try:
       word_file = open (fileName, "r", encoding='utf-8')
       print("file object created")
       window['-TEXT1-'].update("File object created "+ fileName)
       for l in word_file:
           users.append(l.replace('\r', '').replace('\n', ''))
       word_file.close()
       return users
    except:
      print("Error in reading " + fileName)
      window['-TEXT1-'].update("Error in reading "+ fileName)
      if len(users)>0:
          print(users[len(users)-1])
      exit()

#TO MARK USERS IN PHRASES 

def markUsers(phrases):
    finArr = []
    for phrase in phrases:
       marked = getUsersFromNN(phrase)
       print(marked)
       window['-TEXT1-'].update(marked)
       finArr.append(marked)
    return finArr    

   
def getUsersFromNN(phrase):
    # create example sentence
    sentence = Sentence(phrase)
    # predict tags and print
    onlyUsers = []
    if model is None:
        print("MODEL IS ZRO")
        return onlyUsers
    model.predict(sentence)
    sent = sentence.to_tagged_string()
    onlusers = sent.split("[")
    #create tuples    
    if len(onlusers)<2:
        return onlyUsers
    tupleArr = onlusers[1].split(",")
    if len(tupleArr)<1:
        return onlyUsers
    for tp in tupleArr: 
        t = tp.split("/")
        if len(t)>1:
            if t[1] == "<unk>" or t[1] == "<unk>]":
                t[1] = "O"
            onlyUsers.append(tuple(t))
    return onlyUsers

#TO STORE MARKED PHRASES IN FINAL FILE

def writeUsersFile(finalArray):
    finStr = ""
    for arr in finalArray:
           finStr = finStr + (' '.join(str(s) for s in arr) + "\n\n") 
    with open(markedPhrasesFileName, 'a') as f:    
        f.write(finStr.encode("utf-8").decode())  


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
                    if str not in allComUsers: 
                        allComUsers.append(str)
                        comUsersCounts.append(1)
                    else: 
                        ind = allComUsers.index(str)
                        if ind>=0:
                            comUsersCounts[ind] = comUsersCounts[ind] + 1                   
        elif wr == False and found == True:
            for y in range(0, len(arr)-2):
                us = list(arr[y])
                if us[1] == "B":
                    if us[0] not in allSimUsers: 
                        allSimUsers.append(us[0])
                        usersCounts.append(1)
                    else: 
                        ind = allSimUsers.index(us[0])
                        if ind>=0:
                            usersCounts[ind] = usersCounts[ind] + 1
                    

def writeUsersStatFile(): 
    simpleF = open("markedSimpleUsersOnlyFullNN1.txt", "a")
    markedPhrasesFile = open("markedPhrasesFullNN1.txt", "a")
    complexPhrasesFile = open("markedComplexUsersOnlyFullNN1.txt", "a")
    userStatistics = open("markedUsersStatisticsFullNN1.txt", "a")
    usersNotInListFile = open("markedUsersOutOfList1.txt", "a")
    for i in range(0, len(allSimUsers)-1):
        tw = allSimUsers[i] + "-" + str(usersCounts[i]) + "\n"
        simpleF.write(tw)
        userStatistics.write(tw)
    for i in range(0, len(allComUsers)-1):
        tw = allComUsers[i] + "-" + str(comUsersCounts[i])+ "\n"
        userStatistics.write(tw)
        complexPhrasesFile.write(tw)
        if len(allComUsers[i])>1 and allComUsers[i] not in usersList:
            usersNotInListFile.write(tw)
            
def printData():
    print(allSimUsers)
    print(allComUsers)
    window['-USSTAT-'].update("Remaining phrases "+str(len(phrases)))
    window['-TEXT1-'].update(allSimUsers)
    window['-TEXT1-'].update(allComUsers)
    

#MAIN SCRIPT

def writeUsers():
    phr = []  
    threadsL.append(1) 
    print("thread added")
    while len(phrases)>1:
        try: 
            sema.acquire()
            if len(phrases)>=10:
                for i in range(0,10):
                    phr.append(phrases[i]) 
            else:
                phr = phrases
            for i in range(0,len(phr)):
                phrases.pop(0)  
            sema.release()   
            finalArray = markUsers(phr)
            getUsersStatistics(finalArray) 
            writeUsersFile(finalArray)     
            printData()
        except:
            print("Exception in writing users")  
            window['-TEXT1-'].update("Exception in writing users")   
    threadsL.pop(0)
    print("THREAD FINISHED " + str(len(threads)))
    if len(threadsL)<1:
        writeUsersStatFile()
        window['-USSTAT-'].update("")
        print("STATISTICS WRITTEN ")
        window['-TEXT1-'].update("STATISTICS WRITTEN in following files: markedSimpleUsersOnlyFullNN1.txt, markedComplexUsersOnlyFullNN1.txt, markedUsersStatisticsFullNN1.txt, markedUsersOutOfList1.txt")
        window['Open Folder'].update(visible=True)
    
        
    
#OPEN COMMAND

def IButton(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0), background_color='#DAE0E6')

#START THREADS

def startThreads():
    window['Open Folder'].update(visible=False)
    threads = []
    for cnt in range(0,10):
        threads.append(Thread(target=writeUsers))
    for ph in threads:
        print("thread start")
        ph.start()
    return threads


#USER INTERFACE

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
    [sg.Text('', key='-USSTAT-', background_color='#DAE0E6', text_color='black'), IButton('Open Folder',visible=False)]    
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
    if event == 'Open Folder': 
        subprocess.Popen(f'explorer {os.path.abspath(os.getcwd())}')
    if event == 'Ok': 
        phrases.clear()
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
        phrases = getPhrasesFromFile(dataFileName, phStart, phNum)
        usersList = getDataFromFile("usersList.txt")
        print('Taken phrases - ' + str(len(phrases)))
        print("Taken users - "+str(len(usersList)))
        window['-TEXT-'].update('Taken phrases - ' + str(len(phrases)) + " Users taken from users list- "+str(len(usersList)))
        startThreads()
    
window.close()


    
