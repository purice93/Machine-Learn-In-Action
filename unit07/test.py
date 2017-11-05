""" 
@author: zoutai
@file: test.py 
@time: 2017/11/04 
@description: 
"""

# 这里记录一个python错误：当两个类出现互相调用时，可能会出现无法调用的情况，如adaboost和boost两个方法互相调用
# 所以，测试运行代码尽量另外单独写，比如test
from unit07.adaboost import *

# # 测试1:基于加权的分类器
# dataMat,lebalMat = loadSimpData()
# # D是这里的核心：D是每一个样本的权重，通过D.T*errArr来得出错误率，除以m是为了归一化
# # 在单层决策树中并没有迭代，但是才后面的提升过程中，主要就是D的不断迭代修改，来进行训练优化
# D = mat(ones((5,1))/5)
# bestStump,minError,bestClass = buildStump(dataMat,lebalMat,D)
# print(bestStump)
# print(minError)
# print(bestClass)

# # 测试2:基于adaboost的弱分类器，多个阈值，单层
# dataMat,lebalMat = loadSimpData()
# weakClassArray = adaboostTrainDS(dataMat,lebalMat,9)
# print(weakClassArray)

# # 测试3：测试一个数据集，看效果
# dataMat,lebalMat = loadSimpData()
# # 这里的30是迭代次数，也是分类器个数。其实3次时，errorRate==0,就够了，此时分类器就是3个
# weakClassArray = adaboostTrainDS(dataMat,lebalMat,30)
# predictResult = adaClassify([[0,0]],weakClassArray) # [[0,0]]这个地方书上写错了
# print("the result is:",predictResult)

# # 测试4-马疝病-预测得病的马是否能存活
# dataMat,lebalMat = loadDataSet('horseColicTraining2.txt')
# classArr = adaboostTrainDS(dataMat,lebalMat,10)
# print(classArr)
# testMat,testLabelMat = loadDataSet('horseColicTest2.txt')
# predictArr10 = adaClassify(testMat,classArr)
# num = len(testMat)
# errorArr = mat(ones((num,1)))
# errorRate = sum(errorArr[predictArr10 != mat(testLabelMat).T])/num
# print(errorRate)
# # 说明一点，这里的数据与第四章不同；另外我的测试结果和书上也不同
# # 测试结果：
# # 1   --   0.367892976589   --   0.283582089552
# # 10  --   0.354515050167   --   0.328358208955


# 画ROC曲线
# 疑问：这个地方实在是没弄懂，这个曲线有什么用？只是用了训练的排序，没有用训练的预测值。
dataMat,lebalMat = loadDataSet('horseColicTraining2.txt')
classArr,sumClass = adaboostTrainDS(dataMat,lebalMat,10)
plotROC(sumClass.T,lebalMat)
