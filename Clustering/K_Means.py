import datetime
from numpy import *
from numpy.ma import sqrt, power
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

# 欧式向量距离公式
def disteclud(veca, vecb):
    return sqrt(sum(power(veca - vecb, 2)))  # 欧式距离

def _init():  # 初始化各类数据
    global rootFN  # 数据源
    global K  # 簇的数量
    global distForm  # 距离公式
    global K_Mean_IterCount#KMean的迭代次数
    global K_Mean_IterTime#KMean的迭代时间(s)
    global K_Center_IterCount
    global K_Center_IterTime
    rootFN = "C:\\Users\\BanTianZhong\\Desktop\\wine\wine.data"
    K = 3
    distForm = disteclud#距离公式
    K_Mean_IterCount = 0
    K_Mean_IterTime = 0
    K_Center_IterCount = 0
    K_Center_IterTime = 0

def get_rootfile():  # 获得源文件路径,不存在则提醒
    try:
        return rootFN
    except KeyError:
        print("Please set root fliename")


def get_K():  # 获得簇的数量,不存在则提醒
    try:
        return K
    except KeyError:
        print("Please set K")


def loadData(fileroot):  # 加载源数据
    dataSet = []
    filedata = open(fileroot)  # 打开文件
    for fline in filedata.readlines():
        strLine = fline.strip().split(',')  # 按逗号分隔每行数据，字符串
        floLine = list(map(float, strLine))  # 数据转换格式
        dataSet.append(floLine)
    return dataSet  # 数据的二维列表表示


def testData(dataSet, j):  # 将数据集中的类型隐去,形成“测试”数据，j为类型属性的下标
    testSet = []
    for i in range(len(dataSet)):
        testSet.append(dataSet[i][:])
        testSet[i][j] = 0
        testSet[i].append(0)  # 为每个点添加与簇中心距离的记录
    return testSet


def randCore(datalist, K):  # 生成随机核心，数量为K，返回核心的坐标列表
    L = len(datalist)
    random.seed(datetime.datetime.now())#设置随机数种子为当前系统时间的毫秒
    numlist = random.sample(range(L), K)#随机核心的小标
    Corelist = []
    for i in range(K):
        Corelist.append(datalist[numlist[i]][:])
        Corelist[i][0]= numlist[i]#顺便把随机到的点的编号记住
    return Corelist


# 数据归一化
def normalized(datalist):
    datalist = mat(datalist)
    s = shape(datalist)
    for i in range(s[1]-1):
        minv = min(datalist[:,i+1])
        maxv = max(datalist[:,i+1])
        for j in range(s[0]):
            datalist[j,i+1]=(datalist[j,i+1]-minv)/(maxv-minv)
    return datalist.tolist()

def K_Means(testData, K, Corelist):  # K-Means算法
    num = len(testData)  # 参与聚簇点个数
    L = len(testData[0])
    testData = mat(testData)
    Core = mat(Corelist)
    flag = True  # 簇中心改变否？
    global K_Mean_IterCount
    global K_Mean_IterTime
    start = time.time()
    while (flag):
        K_Mean_IterCount += 1  # 迭代次数+1
        flag = False
        for i in range(num):  # 计算每个点与簇中心的距离
            mD = inf  # 最小距离记录
            mI = -1  # 与第标记为mI的簇中心距离最近
            for j in range(K):  # 计算与每个中心的距离
                tempD = distForm(Core[j, 1:], testData[i, 1:L - 1])
                if (tempD < mD):
                    mD = tempD#改变最小距离
                    mI = j + 1#改变簇编号
            if (testData[i, 0] != mI):
                flag = True  # 若不等则改变标志量
            testData[i, 0] = mI
            testData[i, L - 1] = pow(mD, 2)

        for k in range(K):  # 计算新的簇中心
            noKPointSet = testData[nonzero(testData[:, 0].A == k + 1)[0], 1:L - 1]
            Core[k, 0] = k + 1
            Core[k, 1:] = mean(noKPointSet, axis=0)
    end = time.time()
    K_Mean_IterTime = end - start
    return Core, [testData[:, 0], testData[:, L - 1]]


def K_Center(testData, K, Corelist):  # K-中心算法
    num = len(testData)  # 参与聚簇点个数
    L = len(testData[0])
    testData = mat(testData)
    Core = mat(Corelist)
    flag = True  # 簇中心改变否？
    global K_Center_IterCount
    global K_Center_IterTime
    start = time.time()
    while (flag):
        K_Center_IterCount += 1  # 迭代次数+1
        flag = False
        for i in range(num):  # 计算每个点与簇中心的距离
            mD = inf  # 最小距离记录
            mI = -1  # 与第标记为mI的簇中心距离最近
            for j in range(K):  # 计算与每个中心的距离
                tempD = distForm(Core[j, 1:], testData[i, 1:L - 1])
                if (tempD < mD):
                    mD = tempD
                    mI = j + 1
            if (testData[i, 0] != mI):
                flag = True  # 若不等则改变标志量
            testData[i, 0] = mI
            testData[i, L - 1] = pow(mD, 2)

        for k in range(K):  # 计算新的簇中心
            idxset = nonzero(testData[:, 0].A == k + 1)[0].tolist()#该簇的所有点的下标
            noKPointSet = testData[idxset,:]#该簇的所有点
            idxset.remove(int(Core[k, 0]))  # 除去中心点的所有点的下标
            dist1 = sum(noKPointSet[:,L-1]) #之前所有点到中心点的距离之和
            dist2 = dist1+1
            tempcoreid = Core[k,0]
            while(idxset and dist2-dist1>0): #若所有点到新的中心点的距离小于之前的则视为改变
                dist2 = 0
                tempcoreid = random.sample(idxset,1)[0]#随机生成下一个中心点
                for i in range(len(idxset)):#计算其他所有点到簇中心的距离
                    dist2 = dist2 + distForm(testData[idxset[i],1:L-1],testData[tempcoreid,1:L-1])
                dist2 =dist2+pow(distForm(Core[k,1:],testData[tempcoreid, 1:L - 1]),2)
                idxset.remove(tempcoreid)
            if(not idxset):
                continue
            # print(nonzero(testData[:, 0].A == k + 1)[0])
            # print(Core[k,0])
            Core[k,1:] = testData[tempcoreid,1:L-1]
            Core[k, 0] = tempcoreid

        end = time.time()
        K_Center_IterTime = end - start
    return Core, [testData[:, 0], testData[:, L - 1]]



def sumF(i, j, testSqulist):  # 计算词频
    sum = [0] * get_K()
    for m in range(i, j):
        for k in range(get_K()):
            if (int(testSqulist[m, 0]) == k + 1):
                sum[k] += 1
    return sum
# 聚类精确度计算
def CalAccur(datalist, testSqulist):
    L = len(datalist)
    count = [0] * get_K()
    j = i = 0
    while (j < get_K()):  # 计算每一类的个数
        while (i < L):
            if (i == L - 1):
                break
            if (datalist[i][0] == datalist[i + 1][0]):
                count[j] += 1
                i += 1
            else:
                count[j] += 1
                i += 1
                j += 1
        count[get_K() - 1] += 1
        j += 1
    sum = [0] * get_K()
    i = 0
    j = count[0]
    k = 0
    while (k < get_K() - 1):  # 计算每一簇的正确点
        sum[k] = sumF(i, j, testSqulist)
        i = i + count[k]
        j = j + count[k + 1]
        k += 1
    sum[k] = sumF(i, j, testSqulist)
    print(sum)
    rightCount = 0
    for i in range(get_K()):
        maxi = max(sum[i])
        rightCount += maxi
        temp = sum[i].index(maxi)
        for i in range(get_K()):#一旦该簇选择该下标，后面的簇就不能选这个了
            sum[i][temp]=0

    return rightCount / L  # 每一簇的正确点点数/总点数作为聚簇算法的精确度

def showStatic(ResStaList):
    print("最高准确率："+str(max(ResStaList[:,1])[0,0]))
    print("最低准确率："+str(min(ResStaList[:,1])[0,0]))
    print("平均准确率："+str(mean(ResStaList[:,1])))
    print("最高迭代次数:"+str(max(ResStaList[:,0])[0,0]))
    print("最低迭代次数："+str(min(ResStaList[:,0])[0,0]))
    print("平均迭代次数："+str(mean(ResStaList[:,0])))


_init()
datalist = loadData(get_rootfile())
datalist = normalized(datalist)
tesdatalist = testData(datalist, 0)
print(tesdatalist)

i = 0
ResStaList =[]
while(i<5):
    print("第"+str(i)+"次迭代情况：")
    resultSet = K_Means(tesdatalist, get_K(), randCore(datalist, get_K()))
    print("K-Means算法的准确率为：", end="")  # 在0.7左右
    Accur  =CalAccur(datalist, resultSet[1][0])
    print(Accur)
    print("K-Means算法迭代次数为：", end="")
    print(K_Mean_IterCount)
    print("K-Means算法迭代时间为：", end="")
    print(K_Mean_IterTime)
    ResStaList.append([K_Mean_IterCount,Accur])
    i += 1
    K_Mean_IterCount=0
    K_Mean_IterTime=0

print(ResStaList)
showStatic(mat(ResStaList))
tips = DataFrame(ResStaList)
tips.rename(columns={0:'IterCount',1:'Accur'},inplace=True)
sns.set(color_codes=True)
g = sns.lmplot(x="IterCount", y="Accur",data=tips)
plt.show()

# j = 0
# ResStaList_1 =[]
# while(j<50):
#     print("第" + str(j) + "次迭代情况：")
#     resultSet1 = K_Center(tesdatalist, get_K(), randCore(datalist, get_K()))
#     print("K-Center算法的准确率为：", end="")
#     Accur = CalAccur(datalist, resultSet1[1][0])
#     print(Accur)
#     print("K-Center算法迭代次数为：", end="")
#     print(K_Center_IterCount)
#     print("K-Center算法迭代时间为：", end="")
#     print(K_Center_IterTime)
#     ResStaList_1.append([K_Center_IterCount, Accur])
#     j += 1
#     K_Center_IterCount = 0
#     K_Center_IterTime = 0
#
# #数据可视化
# print(ResStaList_1)
# showStatic(mat(ResStaList_1))
# tips = DataFrame(ResStaList_1)
# tips.rename(columns={0:'IterCount',1:'Accur'},inplace=True)
# sns.set(color_codes=True)
# g = sns.lmplot(x="IterCount", y="Accur",data=tips)
# plt.show()