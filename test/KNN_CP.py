from numpy import *
import operator
from os import listdir

def createdataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B'];
    return group,lables


def classify(inX,dataSet,lables,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqdiffMat = diffMat**2
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort();
    classCount = {}
    for i in range(k):
        votelable = lables[ sortedDistIndices[i] ]
        classCount[votelable] = classCount.get(votelable,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

def classify0(inX, dataSet, labels, k):#inX 用于分类输入向量,dataSet输入的训练样本集,lables用于分类的标签,k 最近邻居的数目
    dataSetSize = dataSet.shape[0] #得到数组的行数，即测试数据的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat = diffMat ** 2#各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)#对应想加，即得到了每一个距离的平方
    distances = sqDistances ** 0.5 #开方，得到距离。
    sortedDistIndicies = distances.argsort() #升序排序
    # 选择距离最小的k个点。
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #获得对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #按照标签从字典中取值，key为voteIlabel，如果没有返回0，如果有就加1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #用集合的第二个数据域来进行降排序
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def file2matrix_CP(filename):
    fr = open(filename)
    file = fr.readlines()
    filenumbers = len(file)
    returnMat = zeros((filenumbers,3))
    classLableVector = []
    index = 0
    for line in file :
        fileline = line.strip()
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLableVector.append(int(listFormLine[-1]))
        index+=1
    return  returnMat,classLableVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def autoNorm_CP(dataSet):
    minValues = dataSet.min(0)
    maxvalues = dataSet.max(0)
    ranges = maxvalues - minValues
    # normDataSet = zeros(shape(dataSet))
    normDataSet = dataSet - tile(minValues,(dataSet.shape[0],1))
    normDataSet = normDataSet/tile(ranges,(dataSet.shape[0],1))
    return normDataSet

def datingClassTest():
    hoRatio = 0.10  # hold out 10%
    datingDataMat, datingLabels = file2matrix_CP('E:\machinelearninginaction\Ch02\datingTestSet2.txt')  # load data setfrom file
    # normMat, ranges, minVals = autoNorm_CP(datingDataMat)
    normMat = autoNorm_CP(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def datingClassTest_CP():
    hoRatio = 0.10 #测试数据比例
    data_mat ,data_labels = file2matrix_CP('E:\machinelearninginaction\Ch02\datingTestSet2.txt')
    data_format_mat = autoNorm_CP(data_mat)
    m = data_format_mat.shape[0]
    numTests = int(m*hoRatio)
    error_count = 0.0
    for i in range(numTests):
        result = classify0(data_format_mat[i,:],data_format_mat[numTests:m,:],data_labels[numTests:m],3)

        if result != data_labels[i]:
            error_count += 1.0;
            # print("the classifier came back with: %d, the real answer is: %d" % (result, data_labels[i]))

    print("the total error rate is: %f" % (error_count / float(numTests)))
    print(error_count)


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('E:\machinelearninginaction\Ch02\dtrainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('E:\machinelearninginaction\Ch02\dtrainingDigits/%s' % fileNameStr)
    testFileList = listdir('E:\machinelearninginaction\Ch02\dtestDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('E:\machinelearninginaction\Ch02\dtestDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr) )
        if (classifierResult != classNumStr): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount )
    print( "\nthe total error rate is: %f" % (errorCount / float(mTest)) )