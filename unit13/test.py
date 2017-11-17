""" 
@author: zoutai
@file: test.py 
@time: 2017/11/16 
@description: 
"""
from unit13.pca import *
import matplotlib
import matplotlib.pyplot as plt

dataMat = loadDataSet('testSet.txt')
lowMat,reconMat = pca(dataMat,1)
# lowMat,reconMat = pca(dataMat,2) #数据本身只有两个特征，因此当选择2时，数据不区分，数据重合没有线
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=90)


# test2-利用PCA对半导体数据进行降维
# 方差百分比=（选取的特征值之和）/(所有的特征值之和)
dataMat = replaceNanWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigVals,eigVector = linalg.eig(mat(covMat))
sortEigVals = sort(eigVals)
eigSum = sum(sortEigVals)
sigValInd = sortEigVals[:-(20+1):-1] # 从倒数第topNfeat+1个，取到导数第一个
print(sigValInd)
print(eigVals) # 通过观察可以发现，大部分特征值都是0，只有少部分有用
x=[]
y=[]
allSum = []
for i in(1,2,3,4,5,6,7,20):
    print(sigValInd[:i])
    addSum = sum(sigValInd[i-1:i])
    x.append(i)
    y.append(10*addSum/eigSum)
    allSum.append(10*sum(sigValInd[0:i])/eigSum)
ax = fig.add_subplot(212)
ax.plot(x,y,marker = 'x')
ax.plot(x,allSum,marker = '+',color='r')
plt.show()