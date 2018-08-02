from unit09 import regTrees
from numpy import *
import os


# test1
# testMat = mat(eye(4))
#
# print(testMat)
#
# print(regTrees.bin_split_data_set(testMat, 1, 0.5))


# test2
# my_data = regTrees.load_data_set("ex00.txt")
# my_mat = mat(my_data)
# regTrees.create_tree(my_mat)
# print(regTrees)


# test3
# my_mat2 = mat(regTrees.load_data_set('ex2.txt'))
# my_tree = regTrees.create_tree(my_mat2,options=(0,1))
# my_mat_test = mat(regTrees.load_data_set('ex2test.txt'))
# regTrees.prune(my_tree, my_mat_test)


# test4
my_mat4 = regTrees.load_data_set('exp2.txt')
tree_4 = regTrees.create_tree(my_mat4, regTrees.model_leaf, regTrees.model_error,(1,10))
print(tree_4)

# test more