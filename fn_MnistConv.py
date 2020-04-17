## 20190319_v1_0_0, 최초 생성

import numpy
from scipy import signal
from Softmax_kjw_v1_0_0 import *
from ReLU_kjw_v1_0_0 import *
from Conv_kjw_v1_0_1 import *
from Pool_kjw_v1_0_0 import *

def MnistConv_kjw(W1, W5, Wo, X, D):
    alpha = 0.01
    beta = 0.95

    momentum1 = numpy.zeros_like(W1)
    momentum5 = numpy.zeros_like(W5)
    momentumo = numpy.zeros_like(Wo)

    N = len(D)

    bsize = 100
    blist = numpy.arange(0, N, bsize)

    for batch in range(len(blist)):
        dW1 = numpy.zeros_like(W1)
        dW5 = numpy.zeros_like(W5)
        dWo = numpy.zeros_like(Wo)

        begin = blist[batch]

        for k in range(begin, begin+bsize):
            # Forward pass = inference
            x = X[k, :, :]
            y1 = Conv_kjw(x, W1)
            y2 = ReLU_kjw(y1)
            y3 = Pool_kjw(y2)

            ## numpy.reshape: 열의 개수가 '1'인 행렬로 값을 재배치
            y4 = numpy.reshape(y3, (-1, 1))

            v5 = numpy.matmul(W5, y4)
            y5 = ReLU_kjw(v5)
            v = numpy.matmul(Wo, y5)
            y = Softmax_kjw(v)

            # one-hot encoding
            d = numpy.zeros((10, 1))
            d[D[k][0]][0] = 1

            # Backpropagation
            e = d - y
            delta = e

            e5 = numpy.matmul(Wo.T, delta) # Hidden(ReLU)
            ## (y5 > 0): 참이면 True==1, 거짓이면 False==0
            delta5 = (y5 > 0) * e5

            e4 = numpy.matmul(W5.T, delta5) #Pooling layer

            e3 = numpy.reshape(e4, y3.shape)

            e2 = numpy.zeros_like(y2)
            W3 = numpy.ones_like(y2)/(2*2)
            for c in range(20):
                ## numpy.kron: 왼쪽 행렬의 값 한 개씩을 오른쪽 행렬의 값 한 개씩에 모두 곱한 값들로 행렬을 만든다.
                ## ex) [1 0]  [1 1]    [1 1 0 0]
                ##     [0 1]  [1 1]    [1 1 0 0]
                ##                     [0 0 1 1]
                ##                     [0 0 1 1]
                e2[:, :, c] = numpy.kron(e3[:, :, c], numpy.ones((2, 2)))*W3[:, :, c]

            delta2 = (y2 > 0)*e2
################################################################################################################20190403
            delta1_x = numpy.zeros_like(W1)
            for c in range(20):
                delta1_x[:, :, c] = signal.convolve2d(x[:, :], numpy.rot90(delta2[:, :, c], 2), 'valid')

            dW1 = dW1 + delta1_x
            dW5 = dW5 + numpy.matmul(delta5, y4.T)
            dWo = dWo + numpy.matmul(delta, y5.T)

        dW1 = dW1/bsize
        dW5 = dW5/bsize
        dWo = dWo/bsize

        momentum1 = alpha*dW1 + beta*momentum1
        W1 = W1 + momentum1

        momentum5 = alpha*dW5 + beta*momentum5
        W5 = W5 + momentum5

        momentumo = alpha*dWo + beta*momentumo
        Wo = Wo + momentumo

    return W1, W5, Wo