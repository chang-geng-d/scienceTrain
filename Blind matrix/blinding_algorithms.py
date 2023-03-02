import numpy as np
import random


def kroneckerdelta(x, y):
    if x == y:
        return 1
    else:
        return 0


def generateN(n):
    print('generate N')
    '''n为第一隐藏层节点数'''
    pi = []
    alpha = []
    for i in range(n):
        pi.append(i + 1)
        alpha.append(i + 1)
    for i in range(n):
        a = int(random.random() * (n - 1) + 0.5)
        b = int(random.random() * (n - 1) + 0.5)
        tmp = pi[a]
        pi[a] = pi[b]
        pi[b] = tmp
    for i in range(n):
        a = int(random.random() * (n - 1) + 0.5)
        b = int(random.random() * (n - 1) + 0.5)
        tmp = alpha[a]
        alpha[a] = alpha[b]
        alpha[b] = tmp
    # print(pi)
    # print(alpha)
    N = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            N[i, j] = alpha[i] * kroneckerdelta(pi[i], j + 1)
    # print(N)
    # print(type(N))
    inverse_N = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for x in range(n):
                if pi[x] == i + 1:
                    break
            inverse_N[i, j] = 1 / alpha[i] * kroneckerdelta(x, j)
            # print(x+1)
    # print(inverse_N)
    # print(np.matmul(N,inverse_N))
    inverse_N = np.mat(N).I
    # print(np.mat(N).I)
    return N, inverse_N


def generateR(n, m):
    print('generate R(n*m)')
    '''n为输入层节点数，m为样本个数'''
    R = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            R[i, j] = random.random() * 100
    print(R)
    return R


def generateB(b, m):
    print('expand b to B')
    '''b为第一隐藏层偏置向量，m为样本个数'''
    raw = b.copy()
    for _ in range(m - 1):
        b = np.c_[b, raw]
    print(b)
    return b


if __name__ == '__main__':
    # generateN(10)
    # generateR(5,6)
    # data = np.matrix([np.random.randint(1, 10, 1) for _ in range(6)])
    # print(data)
    # generateB(data, 10)
    pass
