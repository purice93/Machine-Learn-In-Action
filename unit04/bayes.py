

'''
机器学习实战-第四章（贝叶斯-朴素贝叶斯）
'''
from numpy import *
from unit04.getYourCityFromWord import *
# 生成语录数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 创建词汇set
def createVocabSet(dataSet):
    vocabSet = set()
    for sentence in dataSet:
        for word in sentence:
            vocabSet.add(word)
    return vocabSet

# 1词集模型
# 将一个句子映射到上面的set集合上，做成向量
def setOfWordsVec(dataVecSet,vocabSet):
    vocSetOfList = []
    # 方向错误，必须以dataVecSet为准
    # for word in vocabSet:
    #     if word in dataVecSet:
    #         vocSetOfList.append(1)
    #     else:
    #         vocSetOfList.append(0)
    for word in dataVecSet:
        if word in vocabSet:
            vocSetOfList.append(1)
        else:
            vocSetOfList.append(0)
    return vocSetOfList

# 2词袋模型
# 函数意义同setOfWordsVec；但是由于set中默认每个单词只出现一次，所以丢失了单词的个数信息；
# 所以这里将是否含有单词0-1改为含有的"个数"
def bagOfWordsVec(dataVecSet,vocabSet):
    vocSetOfList = [0] * len(dataVecSet)
    for word in dataVecSet:
        if word in vocabSet:
            vocSetOfList[list(dataVecSet).index(word)] += 1
    return vocSetOfList


# 计算每个类别各个特征占总特征数的比值
def trainNB0(trainMatrix,classvec):
    rowLength = len(trainMatrix) # 行数
    print(len(trainMatrix[0]))
    colLength = len(trainMatrix[0])     # 列数
    pAbusive = sum(classvec) / float(rowLength) # 当classVec == 1时，代表是侮辱，计算侮辱性词语的占比（3/6）
    print(pAbusive)
    # p0num = list(zeros(rowLength));p1num  = list(zeros(rowLength))  # 全1， #避免一个概率值为0,最后的乘积也为0
    p0num = list(ones(colLength));p1num  = list(ones(colLength)) # 行向量，表示分为1类，每个特征出现的次数
    p0Denom = 0.0;p1Denom = 0.0 # 分母，所有分为1类，且包含特征单词的个数总数
    for i in range(rowLength):
        if classvec[i] == 1:
            p0num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        else:
            p1num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
    prop0 =p0num/p0Denom
    prop1 = p1num/p1Denom
    return pAbusive,prop0,prop1
    


# 分类函数(贝叶斯公式)
def classifyNB(vec2Classify, p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p2 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1 > p2:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabSet(listOPosts) # 创建一个不重复单词的set
    trainMat = []
    # 将训练数据每一行转换为0-1表示
    for everyRow in listOPosts:
        trainMat.append(setOfWordsVec(myVocabList,everyRow))
    pAb,p0V,p1V = trainNB0(array(trainMat),array(listClasses))

    # 进行举例测试
    testEntry = ['love','my','dalmation']
    testVec = setOfWordsVec(myVocabList,testEntry)
    print(testEntry,'is classified as :',classifyNB(testVec,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    testVec = setOfWordsVec(myVocabList,testEntry)
    print(testEntry,'is classified as :',classifyNB(testVec,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    testVec = bagOfWordsVec(myVocabList,testEntry)
    print(testEntry,'is classified as :',classifyNB(testVec,p0V,p1V,pAb))

# 分别得到两个城市的词频
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = [];topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        else:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])