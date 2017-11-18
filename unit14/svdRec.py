""" 
@author: zoutai
@file: svdRec.py 
@time: 2017/11/17 
@description: 
"""
from numpy import *
def loadExData():
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],]

def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

def loadExData3():
    return[
           [4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

# 三个相似度计算函数：欧氏距离、余弦函数、皮尔逊相关系数
# 量度在0-1之间，0表示相似度最小，反之越大
def eulidSim(inA,inB):
    return 1.0/(1.0 + linalg.norm(inA,inB))

def cosSim(inA,inB):
    num = float(inA.T*inB) # 分子Numerator，分母denominator
    denom = linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)

# 这个内部逻辑不是很清楚
def pearsSim(inA,inB):
    if len(inA) < 3:
        return  1.0
    return 0.5 + 0.5 * corrcoef(inA,inB,rowvar=0)[0][1]

#第一个推荐引擎，不使用SVD，单纯的相似度计算公式来衡量
# standEst对没有进行评分的物品，进行评分的策略
def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0; rateSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        # 寻找对Item和j项商品都进行评分了的所有用户ID
        overLap = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0.0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        simTotal += similarity
        rateSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return rateSimTotal/simTotal

# estMethod=standEst：传递一个函数
def recommend(dataMat,user,N=3,sigMeas=cosSim,estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] # 查找当前user的未评级商品id
    if len(unratedItems) == 0:
        print("you rate allItems")
    itemScores = []
    for item in unratedItems:
        iScore = estMethod(dataMat,user,sigMeas,item)
        itemScores.append((item,iScore))
    # 寻找前N项商品，逆序排序；
    # 疑问：这个jj啥意思？我想应该是：
    # jj[1]取的是分数，即按分数来进行排序，取分数最高的N项
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

# 基于SVD的评分估计-取代standEst
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0; rateSimTotal = 0.0

    # 进行奇异值SVD分解
    U,sigma,VT = linalg.svd(dataMat)
    sigma4 = mat(eye(4)*sigma[:4])


    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        # 寻找对Item和j项商品都进行评分了的所有用户ID
        similarity = simMeas(dataMat[item,].T,dataMat[j,].T)
        simTotal += similarity
        rateSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return rateSimTotal/simTotal

def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1)
            else:
                print(0)
        print('')

# 其实就是一个分解在组合的过程
def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newLine = []
        for i in range(32):
            newLine.append(int(line[i]))
        myl.append(newLine)
    myMat = mat(myl)
    print("原始矩阵：")
    print(myMat,thresh)
    U,sigma,VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k] = sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("压缩后")
    printMat(reconMat,thresh)