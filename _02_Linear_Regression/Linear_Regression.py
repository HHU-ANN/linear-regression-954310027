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
    def down(X, y, w, index, lambdas=0.1):
        # 展开后的二次项的系数之和
        aa = 0
        # 展开后的一次项的系数之和
        ab = 0
        for i in range(X.shape[0]):
            # 括号内一次项的系数
            a = X[i][index]
            # 括号内常数项的系数
            b = X[i][:].dot(w) - a * w[index] - y[i]
            # 可以很容易的得到展开后的二次项的系数为括号内一次项的系数平方的和
            aa = aa + a * a
            # 可以很容易的得到展开后的一次项的系数为括号内一次项的系数乘以括号内常数项的和
            ab = ab + a * b
        # 由于是一元二次函数，当导数为零时，函数值最小值，只需要关注二次项系数、一次项系数和 λ
        return det(aa, ab, lambdas)

    def det(aa, ab, lambdas=0.1):
        w = - (2 * ab + lambdas) / (2 * aa)
        if w < 0:
            w = - (2 * ab - lambdas) / (2 * aa)
            if w > 0:
                w = 0
        return w

    """
            Lasso回归，使用坐标下降法（coordinate descent）
            args:
                X - 训练数据集
                y - 目标标签值
                lambdas - 惩罚项系数
                max_iter - 最大迭代次数
                tol - 变化量容忍值
            return:
                w - 权重系数
            """
    # 初始化 w 为零向量
    w = np.zeros(X.shape[1])
    for it in range(max_iter):
        done = True
        # 遍历所有自变量
        for i in range(0, len(w)):
            # 记录上一轮系数
            weight = w[i]
            # 求出当前条件下的最佳系数
            w[i] = down(X, y, w, i, lambdas)
            # 当其中一个系数变化量未到达其容忍值，继续循环
            if (np.abs(weight - w[i]) > tol):
                done = False
        # 所有系数都变化不大时，结束循环
        if (done):
            break
    return w


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


def main():
    x, y = read_data()
    weight1 = ridge(x, y)
    #weight2 = lasso(x, y)
    return weight1