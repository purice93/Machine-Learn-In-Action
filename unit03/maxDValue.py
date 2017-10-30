# 一个无序数组，求出有序后，相邻两个数的最大值
def index(number,min,step):
    return int((number-min)/step)

a=[0,6,3,16,7,10,9,11,20,18]
maxVal=max(a)
minVal=min(a)
step=(maxVal-minVal)*1.0/len(a)
tongs=[]
for i in range(len(a)+1):
    tongs.append([])
for num in a:
    tongs[index(num,minVal,step)].append(num)
findEmpty=False
serch=False
maxChoice=[]
for i in range(len(a)+1):
    print (tongs[i])
    if len(tongs[i])==0:
        findEmpty=True
        if serch==False:
            serch=True #首次发现空桶，有目标了
            tempMin=max(tongs[i-1])
    else:
        if serch==True:
            tempMax=min(tongs[i])
            maxChoice.append(tempMax-tempMin)
            serch=False
            findEmpty=False #寻找下一个起始空桶
print (maxChoice)
print (max(maxChoice))