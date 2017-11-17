""" 
@author: zoutai
@file: test.py 
@time: 2017/11/13 
@description: 
"""
from unit06.svmMLiA import *
dataArr,labelArr = loadDataSet('testSet.txt')
# b,alpha = smoSimple(dataArr,labelArr,0.6,0.001,40)
# print(alpha[alpha>0])

# 太难，暂时先熟悉玩理论，代码一脸懵逼。。

# test2-p103
# b,alpha = smoP(dataArr,labelArr,0.6,0.001,40)
# print(b)

# test3-p109
# testRbf()

# test4-p113
testDigits(('rbf',20))