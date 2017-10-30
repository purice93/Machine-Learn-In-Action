# 另外我发现一个问题：党创建一个python项目时，会自动创建一个__init__.py文件，
# 这是默认初始化文件，如果直接运行这个文件，首先初始化会运行一次这个文件，之后会再次运行这个文件。也就是这个文件被运行了两次。
from unit11 import Apriori # 这个按照格式来，否则会报错，虽然错误不影响内部逻辑

dataSet = Apriori.loadDataSet2()
print(dataSet)

C1 = Apriori.createC1(dataSet)
print(C1)

D = map(set,dataSet)
# print(list(D))
# C = map(set,C1)
C = map(frozenset,C1)
# print(list(C))

# 注意通过list(map)来将map转为list，只能第一次有效，第二次返回的list为空（可能是由于map是迭代的，一次list遍历后，就到了末尾）
data1 = list(D)
label = list(C)
L1,suppDataMap = Apriori.scanD(data1,label,0.5)
print('满足的支持度元素是：'+str(L1))
print('所有元素的支持度是：'+str(suppDataMap))
print('--------p207')

L,suppDataMap2 = Apriori.apriori(data1)
for onel in L:
    print(onel)
    print(str(suppDataMap2))

print('-----11')
rules = Apriori.generateRules(L,suppDataMap2,0.5)
print(rules)

print('发现毒蘑菇的相似特征')
# 毒蘑菇分类
mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
L,supportMapMush = Apriori.apriori(mushDataSet,0.3)

# 其中L为关联集合标签，标签中2代表是毒蘑菇，所以只需要判断那些标签含有2，就可以大体确定那些可能是毒蘑菇
# 注意数据中的数字代表特征，所有的特征从1开始标，不能重复
for row in L:
    for item in row:
        # set.intersection(item)
        # Return a new set with elements common to the set and all others.
        # 即返回一个包含此元素的新的set
        if item.intersection('2'):
            print(item)
