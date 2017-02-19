from numpy import *
import operator

'''得到测试数据集和分类标签'''
def file2matrix(filename):
    fr = open(filename) #打开文件
    arraylines = fr.readlines();#读取所有行的文件
    numbersofline = len(arraylines);#获得文件行数
    returnMatrix = zeros((numbersofline,3));#创建相应的矩阵
    classLabelVector =[];#存放特征的向量
    index = 0;
    for line in arraylines:#遍历每一行文件
        line = line.strip();
        listsFormLine = line.split("\t");
        returnMatrix[index,:] = listsFormLine[:3];
        classLabelVector.append(listsFormLine[-1]);
        index += 1;
    return returnMatrix,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) #取每一列最小值，0表示按列取，1表示按行取
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals; #得到每一列，最大值和最小值的差
    m = dataSet.shape[0] #dataset.shape 得到数据集的范围，0代表第一维即行数
    dataSet = dataSet - tile(minVals,(m,1)) #得到每个数据与最小值之间的差值
    dataSet = dataSet / tile(ranges,(m,1)) #归一化数据
    return  dataSet

def classify(inX,dataSet,labels,K):
    m = dataSet.shape[0]
    diffMat = dataSet - tile(inX,(m,1))
    sqDiffMat = diffMat**2  #获得平方和
    sumDiffMat = sqDiffMat.sum(axis=1) #按行求和
    distenceMat = sumDiffMat**0.5
    sortedDistence = distenceMat.argsort(); #排序 返回排序后元素在原来数组中的位置索引 It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
    classCount = {} #利用字典来存放数据

    for i in range(K):#遍历前K个数据
        voteLabel = labels[sortedDistence[i]]; #获得原数据标签
        classCount[voteLabel] = classCount.get(voteLabel,0)+1; #统计标签出现次数

    ''' sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list
        iterable：是可迭代类型;
        cmp：用于比较的函数，比较什么由key决定,有默认值，迭代集合中的一项;
        key：用列表元素的某个属性,用于进行比较，有默认值，迭代集合中的一项;
        reverse：排序规则. reverse = True 或者 reverse = False，有默认值。
        返回值：是一个经过排序的可迭代类型，与iterable一样
    '''
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True); #对数据进行排序

    return sortedClassCount[0][0] #返回类别标签

'''综合测试函数'''
def improveBlindDate(filename,ratio,k):
    dataMat,labels = file2matrix(filename) #调函数读文件
    normDataSet = autoNorm(dataMat)#数据集归一化
    arrayLines = normDataSet.shape[0]#获得文件行数
    testLines = int(arrayLines*ratio)#测试文件的行数
    errorCount = 0.0
    for i in range(testLines):
#        results = classify(normDataSet[i],normDataSet[testLines:arrayLines],labels,k)
        results = classify(normDataSet[i,:], normDataSet[testLines:arrayLines,:], labels[testLines:arrayLines], k)
        if results!= labels[i] :  errorCount += 1;

    print(errorCount)
    print(errorCount/testLines)