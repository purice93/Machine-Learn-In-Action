""" 
@author: zoutai
@file: boost.py
@date: 2017/11/04
@description: 单层决策树-找出最好的一个特征，且最好的中间阈值;
"""
from numpy import *
from unit07.adaboost import *

# 这个函数的作用是：对第dimen个特征进行分类，样本点dimen特征小于阈值的被标记为-1，否则标记为+1,。
# 输出为M*1的二维数组，相当于把每个样本当做只有一个特征来划分。这里的1、-1对应标签的1、-1，后面会比较。
# 注意：这里需要对python特别熟悉，shape(dataMat)[0] == m；
# dataMat[:,dimen] <= threshVal返回的是第一列下标，逻辑复杂但是表达却相当简化，这也许就是python的魅力吧
def stumpClassify(dataMat,dimen,threshVal,threshInequal):
    sampleArray = ones((shape(dataMat)[0],1))
    if threshInequal == 'lt':
        sampleArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        sampleArray[dataMat[:,dimen] > threshVal] = 1.0
    return sampleArray

def buildStump(dataArr, classLabels,D):
    dataMat = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMat)
    numSteps = 10;
    bestStump = {}; # 用于保存重要的结果值，如阈值，样本中节点下标等。
    bestClass = mat(zeros((m,1))) # 分类结果mat，与sampleArray对应
    minError = inf
    # 遍历特征变量
    for i in range(n):
        # 找出第i个特征值的最小值和最大值
        rangeMin = dataMat[:,i].min();rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps # 每一步特征值间隔

        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                dimenPredictedVals = stumpClassify(dataMat,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[dimenPredictedVals == labelMat] = 0 # 正确分类的设为0，错误的默认为1
                # 重点来了，这里才是提升树的核心：通过D来调整特征值对应权值w的比重。D第一次需要初始化
                weightError = D.T*errArr
                print("split: dim %d,thresh %.2f,hresh inequal: %s,the weighted error is %.3f" % (i,threshVal,inequal,weightError))

                if weightError < minError:
                    minError = weightError
                    bestClass = dimenPredictedVals.copy() # 会被修改，所以需要copy
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClass

