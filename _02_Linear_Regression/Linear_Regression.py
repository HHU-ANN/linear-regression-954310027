# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os


try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(xArr, yArr):
    def ridgeTest(xMat, yMat, lam=0.2):
        xTx = xMat.T * xMat
        denom = xTx + np.eye(np.shape(xMat)[1]) * lam
        if np.linalg.det(denom) == 0.0:
            return
        ws = denom.I * (xMat.T * yMat)
        return ws

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)  # 数据标准化
    # print(yMean)
    yMat = yMat - yMean
    # print(xMat)
    # regularize X's
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar  # （特征-均值）/方差
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):  # 测试不同的lambda取值，获得系数
        ws = ridgeTest(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def lasso(X, y, lambdas=0.1, max_iter=1000, tol=1e-4):
    pass


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


def main(data):
    x, y = read_data()
    weight1 = ridge(x, y)
    #weight2 = lasso(x, y)
    return weight1