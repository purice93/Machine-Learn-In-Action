

'''
机器学习实战-第四章（贝叶斯）
'''
from numpy import *
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

# 将一个句子映射到上面的set集合上，做成向量
def vecOfDataSet(dataList,vocabSet):
    vocSetOfList = []
    for word in vocabSet:
        if word in dataList:
            vocSetOfList.append(1)
        else:
            vocSetOfList.append(0)
    return vocSetOfList

# 计算每个类别各个特征占总特征数的比值
def trainNB0(trainMatrix,classvec):
    rowLength = len(trainMatrix) # 行数
    print(len(trainMatrix[0]))
    colLength = len(trainMatrix[0])     # 列数
    pAbusive = sum(classvec) / float(rowLength) # 当classVec == 1时，代表是侮辱，计算侮辱性词语的占比（3/6）
    print(pAbusive)
    # p0num = list(zeros(rowLength));p1num  = list(zeros(rowLength))  # 全1， #避免一个概率值为0,最后的乘积也为0
    p0num = list(ones(rowLength));p1num  = list(ones(rowLength)) # 行向量，表示分为1类，每个特征出现的次数
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
    

postingList,classVec = loadDataSet()
vocabSet = createVocabSet(postingList)
print(vocabSet)
vocSetOfList = vecOfDataSet(postingList[0],vocabSet)
print(vocSetOfList)

trainMat = []
for rowlist in postingList:
    rowlistset = vecOfDataSet(rowlist,vocabSet)
    trainMat.append(rowlistset)
print(trainMat)
print(classVec)

pA,p0,p1 = trainNB0(trainMat, classVec)
print(p0)
print(p1)

# 分类函数
def classifyNB(vec2Classify, p0Vec,p1Vec,pClass1):

def