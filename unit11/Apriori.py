
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def loadDataSet2():
    return [[1,3,4,6],[2,3,5,6],[1,2,3,5,6],[2,5,6]]


# 得出数据集的单个元素Set
def createC1(dataSet):
    """
    :param dataSet:
    :return:
    """
    C1 = []
    for data in dataSet:
        for item in data:
            if not [item] in C1: #需要通过[]将元素封装为元组，便于后续作为一个整体进行集合运算
                C1.append([item])
    return C1

# 计算单个元素的支持度，即占比
def scanD(data, Ck, minSupport):
    """
    相当于遍历D，求出所有的Ck中的元素支持度大于minsupport的
    :return: 满足支持度的元素、所有元素-支持度的map
    :param data: dataSet
    :param Ck:元素集合
    :param minSupport:最小支持度阈值
    """
    eleMap = {} # 元素-元素总个数 的映射
    for oneData in data:
        for element in Ck:
            if element.issubset(oneData): # set子集
                # 这样写会报错，因为字典的key不可变，而set是可变的，不能作为字典{}的key，所以需要使用frozenset(冻结set，不可变)
                # TypeError: unhashable type: 'set' --> element
                if not eleMap.__contains__(element): # 这里与原书不同
                    eleMap[element] = 1
                else:
                    eleMap[element] += 1

    # 计算元素支持度
    # 注意：错误object of type 'map' has no len()
    # 原因：In Python 3, map returns an iterator not a list:
    # python3中map是一个迭代器，不能使用len(),需要将map转化为list才能使用
    # numsum = float(len(data))
    numsum = len(list(data)) # 这个地方报错是因为在其它代码处，使用了list作为变量！记住：永远不要使用关键词作为变量
    supportMap = {} # 所有元素支持度的值
    retList = [] # 满足支持度的元素
    for key in eleMap:
        try:
            support = eleMap[key]/float(numsum)
        except:
            support = 0.0
        if support >= minSupport:
            retList.insert(0,key) # 将key插入到第0个位置之前（即最开始位置）
        supportMap[key] = support
    return retList,supportMap

# 将组合Lk合称为每个集合为k个元素的集合
# 如[{1},{2},{3}]-->[{1,2},{2,3},{1,3}]
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        # 注意：下面的逻辑复杂，需要参考书中p208
        # k-2,是为了将前k-2项相同时，集合合并；比如当k=2时，只有一个元素{1}、{2}，此时k-2=0，L1==L2,直接并集运算
        # k=3时，有两个元素{1,2}、{1,3}、{2,3}、{5,6}；由于只能合并为大小为k=3的集合，所以必然只有k-2项是相同的，也只需要组合这种情况
        # 此时组合只有{1,2,3}；
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            # print '-----i=', i, k-2, Lk, Lk[i], list(Lk[i])[: k-2]
            # print '-----j=', j, k-2, Lk, Lk[j], list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            # if first k-2 elements are equal
            if L1 == L2:
                # set union
                # print 'union=', Lk[i] | Lk[j], Lk[i], Lk[j]
                retList.append(Lk[i] | Lk[j])
    return retList

# 提升一级：计算所有可能集合的支持度
# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataSet, minSupport=0.5):
    """apriori（首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）
    Args:
        dataSet 原始数据集
        minSupport 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    C1 = list(map(frozenset,C1))
    # print 'C1: ', C1
    # 对每一行进行 set 转换，然后存放到集合中
    D = list(map(set, dataSet))
    # print 'D=', D
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)
    # print "L1=", L1, "\n", "outcome: ", supportData

    # L 加了一层 list, L 一共 2 层 list
    L = [L1]
    k = 2 # 一组合有两个元素为起点
    # 判断 L 的第 k-2 项的数据长度是否 > 0。第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k-2]) > 0):
        # print 'k=', k, L, L[k-2]
        # 输出k个元素组合时，所对应的集合，并求解对应满足的支持度
        Ck = aprioriGen(L[k-2], k) # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        # print 'Ck', Ck

        Lk, supK = scanD(D, Ck, minSupport) # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK) # 理解update的含义
        # 下面是为了排除最后一个[]空值
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
        # print 'k=', k, len(L[k-2])
    return L, supportData

# 输出关联项及对应的值；如p->h
def generateRules(L,supportData,minConf):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] # 这里的H1是freqSet的变形，起初看了半天没明白
            if i==1:
                calcCong(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                ruleFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcCong(freqSet,H,supportData,bigRuleList,minConf=0.7):
    prunedH = [] # 单词pruned-修剪
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'is:',conf)
            bigRuleList.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

# 这个是为了解决另一个情况，例如：{2，3，4}，我们需要找{2,3}-->{4},还需要知道{2}-->{3,4},即hmp1={2,3},m=2
def ruleFromConseq(freqSet,H,supportData,bigRuleList,minConf=0.7):
    length = len(H[0])
    if len(freqSet) > (length+1):
        sonH = aprioriGen(H,length+1)
        isSonH = calcCong(freqSet,sonH,supportData,bigRuleList,minConf)
        # 这里使用了一个数学逻辑，如果isSonH只有一个，就说明无法再划分
        #（定理：如果某条规则不满足可信度，左部所有的子集也不满足可信度）
        if len(isSonH) > 1: # 去掉这个语句也行，只是计算量增大
            ruleFromConseq(freqSet,isSonH,supportData,bigRuleList,minConf)