""" 
@author: zoutai
@file: pca.py 
@time: 2017/11/16 
@description: PCA降维
"""

from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0) # 计算协方差矩阵
    eigVal,eigVector = linalg.eig(covMat) # 求协方差矩阵的特征值和特征向量

    # 对特征值进行排序
    sortEigVals = argsort(eigVal)
    sigValInd = sortEigVals[:-(topNfeat+1):-1] # 从倒数第topNfeat+1个，取到导数第一个
    selectedEigVertor = eigVector[:,sigValInd]

    # 还原数据
    lowDataMat = meanRemoved * selectedEigVertor
    reconMat = lowDataMat * selectedEigVertor.T + meanVals
    return lowDataMat,reconMat

# 补全缺失值
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
