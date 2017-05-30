# -*- coding: utf-8 -*-
"""
Created on Fri May 26 01:47:48 2017

@author: Harsha SlimShady
"""

from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],['maybe', 'not', 'take', 'him','to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute','I', 'love', 'him'],['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return vocabSet

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[list(vocabList).index(word)] = 1
        else: 
            print("%s is not in my vocabulary"%word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Denom = 2.0
    p1Denom = 2.0
    p0Num = ones(numWords);p1Num = ones(numWords)
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p0>p1:
        return 0
    else:
        return 1

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for doc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,doc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print('test case classified as: %d'%classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print('test case classified as: %d'%classifyNB(thisDoc,p0V,p1V,pAb))
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[list(vocabList).index(word)] += 1
    return returnVec

def textParse(bigString):
    import re 
    regEx = re.compile('\\W*')
    listOfTockens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTockens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pAb = trainNB0(array(trainingMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        x1 = classifyNB(array(wordVector),p0V,p1V,pAb)
        x2 = classList[docIndex]
        print('classifier says %d, actually is %d'%(x1,x2))
        if x1 != x2 : errorCount +=1
    print('The error rate is: ',float(errorCount)/len(testSet))

def calMostFreq(vocabList,fullText):
    import operator
    freeDict = {}
    for token in vocabList:
        freeDict[token] = fullText.count(token)
    sortedFreq = sorted(freeDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse0(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse0(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def textParse0(bigString):
    x = bigString.strip(',')
    import re 
    regEx = re.compile('\\W*')
    listOfTockens = regEx.split(x)
    return [tok.lower() for tok in listOfTockens if len(tok)>2]





