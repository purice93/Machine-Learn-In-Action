from unit09 import regTrees
from numpy import *
import os


testMat = mat(eye(4))
# print(testMat)
#
# print(regTrees.bin_split_data_set(testMat, 1, 0.5))
# print(testMat.T.tolist()[0])

# path=os.path.abspath('.')


my_data = regTrees.load_data_set("ex00.txt")
my_mat = mat(my_data)
regTrees.create_tree(my_mat)
print(regTrees)
