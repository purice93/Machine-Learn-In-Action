from numpy import *


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


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


# 剪枝
def prune(tree, test_data):
    if shape(test_data)[0] == 0:
        return get_mean(tree)
    if (is_tree(tree['right'])) or (is_tree(tree['left'])):
        left_set, right_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        prune(tree['left'], left_set)
    if is_tree(tree['right']):
        prune(tree['right'], right_set)

    if (not is_tree(tree['right'])) and (not is_tree(tree['left'])):
        left_set, right_set = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        error_no_merge = sum(pow(left_set - tree['left'], 2)) + sum(pow(right_set - tree['right'], 2))
        tree_mean = (get_mean(tree['left']) + get_mean(tree['right'])) / 2.0
        error_merge = sum(pow(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print("merging...")
            return tree_mean
        else:
            return tree
    else:
        return tree


# 线性回归方程，y = w * x，求解w
# 最小二乘法
def linear_solve(data_set):
    m, n = shape(data_set)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1, n] = data_set[:, 0, n - 1]
    Y = data_set[:, -1]
    xtx = X.T * X
    if linalg.det(xtx) == 0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    ws = xtx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(data_set):
    ws, X, Y = linear_solve(data_set)


def model_error(data_set):
    ws, X, Y = linear_solve(data_set)
    y_hat = ws * X
    return sum(pow(y_hat - Y), 2)


def reg_tree_eval(model, input_data):
    return float(model)


def model_tree_eval(model, input_data):
    n = shape(input_data)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = input_data
    return float(X * model)


def treeForeCast(tree, inData, model_eval=reg_tree_eval):
    if not is_tree(tree): return model_eval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return treeForeCast(tree['left'], inData, model_eval)
        else:
            return model_eval(tree['left'], inData)
    else:
        if is_tree(tree['right']):
            return treeForeCast(tree['right'], inData, model_eval)
        else:
            return model_eval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=reg_tree_eval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat