""" 
@author: zoutai
@file: test.py 
@time: 2017/11/21 
@description: 
"""
from unit10.kMeans import *
from unit10.plotTest import *
from numpy import *

# test1
dataMat = mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing = kMeans(dataMat,4)
print(myCentroids)

# test2
# dataMat3 = mat(loadDataSet('testSet2.txt'))
# centList,newAssment = biKmeans(dataMat3,3)
# print(centList)
# plotGraph(dataMat3,centList)
# # dataLast = row_stack(dataMat3,mat(centList))
# print(centList)


# test3
clusterClubs(5)
