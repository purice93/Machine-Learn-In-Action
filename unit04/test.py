from unit04.bayes import *


postingList,classVec = loadDataSet()
vocabSet = createVocabSet(postingList)
print(vocabSet)
vocSetOfList = setOfWordsVec(postingList[0],vocabSet)
print(vocSetOfList)

trainMat = []
for rowlist in postingList:
    rowlistset = setOfWordsVec(rowlist,vocabSet)
    trainMat.append(rowlistset)
print(trainMat)
print(classVec)

pA,p0,p1 = trainNB0(trainMat, classVec)
print(p0)
print(p1)

print("test P64----classify")
testingNB() # 我的测试结果是the error rate is : 0.6，一直在这左右徘徊。不知道为什么书上的误差会这么底，我的可能错了？