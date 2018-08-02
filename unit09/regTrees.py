from numpy import *
import pdb


def load_data_set(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t');
        # map的用法在python2和3中有所不同，需要添加list转换
        float_line = list(map(float, cur_line))
        data_mat.append(float_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):
    mat0 = data_set[nonzero(data_set[:, feature] > value)[0], :]
    mat1 = data_set[nonzero(data_set[:, feature] <= value)[0], :]
    return mat0, mat1


# 左节点：默认为均值
def reg_leaf(data_set):
    return mean(data_set[:, -1])


# 均方总误差
def reg_err(data_set):
    return var(data_set[:, -1]) * shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, error_type=reg_err, options=(1, 4)):
    with_err = options[0];
    with_feature = options[1]
    # 如果所有的特征值都相同，则返回
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = shape(data_set)
    err = error_type(data_set)

    best_error = float('inf')
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        # python中一个matrix矩阵名.A 代表将 矩阵转化为array数组类型
        for split_value in set(data_set[:, feat_index].T.A.tolist()[0]):
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_value)
            if (shape(mat0)[0] < with_feature) or (shape(mat1)[0] < with_feature):
                continue
            new_total_err = error_type(mat0) + error_type(mat1)
            if new_total_err < best_error:
                best_index = feat_index
                best_value = split_value
                best_error = new_total_err

    if (err - best_error) < with_err:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)
    if (shape(mat0)[0] < with_feature) or (shape(mat1)[0] < with_feature):
        return None, leaf_type(data_set)
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, error_type=reg_err, options=(1, 4)):
    feat, val = choose_best_split(data_set, leaf_type, error_type, options)
    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val

    l_set, r_set = bin_split_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, error_type, options)
    ret_tree['right'] = create_tree(r_set, leaf_type, error_type, options)
    return ret_tree
