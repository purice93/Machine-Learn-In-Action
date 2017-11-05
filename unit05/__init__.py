# 出现一些微小的错误，没事迷茫，应该是几个函数库的不同所引起的，比如random函数，math、numpy中的都有，
# 可能有一些区别，但是书上是用python2写的，和python3有些区别

from unit05 import logRegres
import numpy as np
dataX,labelY = logRegres.loadDataSet()

# weights1 = logRegres.gradAscent(dataX,labelY)
# logRegres.plotBestFit(weights1)

# weights2 = logRegres.randomGradAscent(dataX,labelY)
# logRegres.plotBestFit(weights2)

# weights3 = logRegres.randomGradAscentPlus(dataX,labelY,150)
# logRegres.plotBestFit(weights3)

logRegres.mutiTest()

# 测试结果:
# the errorRate is: 0.701493
# after 10 itertions,the errorRate mean is 0.661194