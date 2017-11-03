
# 案例3：使用beyes从个人广告中获取区域倾向
# 书上说的有点婉转，使得不能很好的和本章联系；其实这里就是给你两个城市的人们说话的用词，
# 然后通过这些用词来区分某个人属于哪个城市，另外这间接的可以得出城市的词云库信息；
# 另外这种方式也是广告商们如何分析用户的行为信息来确定用户的年龄和职业来进行精确推送；还有一些婚恋网站匹配等
from unit04.filterEmail import *
import feedparser

# 这里使用了词袋模型
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.cout(word)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter,reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabSet(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    # 去掉出现频率最高的那些词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainSet = list(range(2 * minLen));
    testSet = []
    for i in range(20):
        import numpy
        randIndex = int(numpy.random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = [];trainClassList = []
    for docIndex in trainSet:
        trainMat.append(bagOfWordsVec(vocabList,docList[docIndex]))
        trainClassList.append(classList[docIndex])
    pSpam,p0V,p1V = trainNB0(array(trainMat),trainClassList)

    errorCount = 0
    for docIndex in testSet:
        testVec = bagOfWordsVec(vocabList,docList[docIndex])
        if classifyNB(testVec,p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the  rate is :',float(errorCount)/len(testSet))
    return pSpam,p0V,p1V

# 测试
# 说明：可能是由于网络访问的原因，这个无法完成
# ny = feedparser.parse('http://newyork.craiglist.org/stp/insex.rss')
# sf = feedparser.parse('http://sfbay.craiglist.org/stp/insex.rss')
ny = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')
vocabList,pSF,pNY = localWords(ny,sf)

# 得到两个城市的词频
getTopWords(ny,sf)