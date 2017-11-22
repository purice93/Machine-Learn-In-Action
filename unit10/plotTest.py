""" 
@author: zoutai
@file: plotTest.py 
@time: 2017/11/22 
@description: 
"""

# 画这个图费了好长时间，主要的问题还是在于各种数据类型的转换和匹配，真是想说python真是个垃圾语言，对初学者很难适应

import matplotlib.pyplot as plt
from numpy import *
def plotGraph(dataSet,centList):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    newDataSet = zeros((len(centList),2))
    newDataSet = [list(array(centList[i])[0]) for i in range(len(centList))]
    # for i in range(len(centList)):
    #     newDataSet.(array(centList[i])[0])
    newMat = mat(newDataSet)
    print(newMat)
    # ax.plot(dataSet[:,0],dataSet[:,1],marker = 'x')
    # ax.plot(centList[:,0],centList[:,1],marker = '+',color='r')
    ax.scatter(dataSet[:,0].flatten().A[0],dataSet[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(newMat[:,0].flatten().A[0],newMat[:,1].flatten().A[0],marker='o',s=90)
    plt.show()


