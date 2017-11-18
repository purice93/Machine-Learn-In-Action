""" 
@author: zoutai
@file: test.py 
@time: 2017/11/17 
@description: 
"""
from numpy import *
from unit14.svdRec import *
set_printoptions(suppress=True) # 设置numpy不以科学计数法输出，即e
# tets1
U,sigma,VT = linalg.svd([[1,1],[7,7]])
#print(U,sigma,VT)


# test2
U,sigma,VT = linalg.svd(loadExData())
# 对角矩阵，但是写成一维数组的形式；[  9.64365076e+00   5.29150262e+00   7.40623935e-16   4.05103551e-16  2.21838243e-32]
print(sigma) # 通过观察sigma发现，前三项远大于后面几项，所以取前三项就能代表所有项

# 重构对角矩阵（前三项）
sigma3 = mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
# 还原数据
newDataMat = U[:,:3] * sigma3 * VT[:3,:]
print(array(newDataMat))


# test3-餐厅菜肴推荐----result:[(2, 2.5), (1, 2.0243290220056256)]
# 再次提醒自己：数据格式，mat，array等的区别，用法
recommendItems = recommend(mat(loadExData3()),2)
print("为第2个用户推荐菜单是：",recommendItems)


# test4-利用SVD提高优化效果----result:[(3, 3.2440521369249682), (10, 3.1576182912586654), (8, 3.1428571428571428)]
recommendItems4 = recommend(mat(loadExData2()),2,estMethod=svdEst)
print("为第2个用户推荐菜单是：",recommendItems4)
# 这个此时对不上号，可能存在错误


# 基于SVD的图像压缩
imgCompress()