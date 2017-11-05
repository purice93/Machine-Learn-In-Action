
from numpy import *
from unit07.boost import *

""" 
@author: zoutai
@file: adaboost.py 
@date: 2017/11/04
@description:
"""

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def adaboostTrainDS(dataArr,classLabels,numIt):
    weakClassArr = [] # 弱分类器
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    sumBestClass = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,bestClass = buildStump(dataArr,classLabels,D)
        print(D.T)
        # 核心：alpha是单个分类器的权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) # 这里max是防止分母为0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print(bestClass.T)
        # 更新每个决策树的权重
        # 以下使用了P117和P118的公式
        expon = multiply(-1.0*alpha*mat(classLabels).T,bestClass) # classLabels*bestClass 相同则为正
        D = multiply(D,exp(expon))
        print("---",sum(D))
        D = D/D.sum()

        # 以下是为了累计计算错误率，当错误率为0时，终止循环。
        # 但是我有个疑问，问什么要乘以alpha？。如果不乘以alpha，后面也不用sign函数，不可以吗？
        sumBestClass += alpha*bestClass
        print(sumBestClass.T)
        sumError = multiply(sign(sumBestClass) != mat(classLabels).T,ones((m,1)))
        errorRate = sum(sumError) / m
        print("total errorRate is : ",errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr,sumBestClass

# 对给定数据，在每一个分类器上都进行分类，然后加权求解
def adaClassify(testData,classArr):
    testMat = mat(testData)
    m = shape(testData)[0]
    sumClassArr = mat(zeros((m,1)))
    for i in range(len(classArr)):
        oneClassArr = stumpClassify(testMat,classArr[i]['dim'],classArr[i]['thresh'],classArr[i]['ineq'])
        sumClassArr += classArr[i]['alpha']*oneClassArr
        print(sumClassArr) # 加权计算每一个分类，
    return sign(sumClassArr)


# 将文本数据转化为矩阵训练数据
def loadDataSet(filename):
    dataMat = []; labelMat = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        line = line.strip().split()
        for i in range(len(line)-1):
            lineArr.append(float(line[i]))
        dataMat.append(lineArr)
        labelMat.append(float(line[-1]))
    return dataMat,labelMat

# ROC曲线绘制及AUC计算函数
def plotROC(sumClass,classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0) # 计算真阳例的分母--真实结果为正的数目
    numNoPosClas = len(classLabels) - numPosClas # 计算假阳例的分母--...
    yStep = 1/float(numPosClas) # y轴上个步进长度，下同理
    xStep = 1/float(numNoPosClas)
    sortedIndicies = argsort(sumClass) # 重点：argsort函数返回的是数组值从小到大的索引值 参考：http://blog.csdn.net/maoersong/article/details/21875705
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 这里理解是一个难点！
    # 1、书上说了，由于排序是由大到小，也就是，开始值小的部分，训练师被分为-1，之后数大的部分被分为+1；所以只能从右上开始画。、
    # 这样，显然开始画图部分是从不预测值不是+1开始的
    # 但是看图时，是从左下开始，预测值更可能是1.0
    # 表达能力有限啊!我自己都说不清楚。。
    # 这个ROC开起来可能好像没用到训练情况，其实，这里的关键在于排序。这样，排序的前面所有都是-1，后面所有都是+1；。
    # 但是为什么没有使用预测结果，因为最终都将是归为x=1,y=1.目的是让+1尽可能靠近前面，面积AUC更大，这就是ROC的意义！
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0: # 这里的1.0是指预测为正，将实际值classLabels与此比较，如果相同，y值下降
            delX = 0;delY = yStep;
        else:
            delX = xStep;delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive Rate');
    plt.ylabel('True positive Rate');
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is :",ySum*xStep)