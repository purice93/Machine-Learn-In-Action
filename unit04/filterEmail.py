# 案例2：过滤垃圾邮件
from unit04.bayes import *
def textParse(bigString):
    import re
    sentence = re.split('\W+',bigString)
    return [word.lower for word in sentence if len(word) > 2] # 去掉长度小于2的单词

def spamTest(): # spam垃圾邮件
    # import os
    # os.chdir(r'E:/JavaEE_IJ_WorkSpace/MLInAction')
    docList = [];fullText = [];classList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabSet(docList)
    trainSet = list(range(50));testSet = []
    for i in range(10):
        import numpy
        randIndex = int(numpy.random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = [];trainClassList = []
    for docIndex in trainSet:
        trainMat.append(setOfWordsVec(vocabList,docList[docIndex]))
        trainClassList.append(classList[docIndex])
    pSpam,p0V,p1V = trainNB0(array(trainMat),trainClassList)

    errorCount = 0
    for docIndex in testSet:
        testVec = setOfWordsVec(vocabList,docList[docIndex])
        if classifyNB(testVec,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is :',float(errorCount)/len(testSet))

spamTest()