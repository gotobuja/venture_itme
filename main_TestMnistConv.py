# File encoding: UTF-8

"""
20190319_v1_0_0, 최초 생성
20200418_V1_0_1, 미연에게 의뢰 받은 후 첫 코드작성
"""



import numpy
from scipy import signal
from LoadMnistData_kjw_v1_0_0 import *
from Softmax_kjw_v1_0_0 import *
from ReLU_kjw_v1_0_0 import *
from Conv_kjw_v1_0_1 import *
from Pool_kjw_v1_0_0 import *
from MnistConv_kjw_v1_0_0 import *

# Learn
#
Images, Labels = LoadMnistData_kjw('MNIST\\t10k-images-idx3-ubyte.gz', 'MNIST\\t10k-labels-idx1-ubyte.gz')

## Images를 255로 나눔, 0 ~ 1사이 값을 만들기 위함인것으로 추정됨
Images = numpy.divide(Images, 255)

## numpy.random.randn: 가우시안 표준 정규 분포로 난수 생성(기대값: 0, 표준편차: 1)
W1 = 1e-2 * numpy.random.randn(9, 9, 20)

## numpy.random.uniform: 균등분포로 난수 생성, numpy.sqrt: 제곱근
W5 = numpy.random.uniform(-1, 1, (100, 2000)) * numpy.sqrt(6) / numpy.sqrt(360 + 2000)
Wo = numpy.random.uniform(-1, 1, (10, 100)) * numpy.sqrt(6) / numpy.sqrt(10 + 100)

X = Images[0:8000, :, :]
D = Labels[0:8000]

for _epoch in range(3):
    print(_epoch)
    W1, W5, Wo = MnistConv_kjw(W1, W5, Wo, X, D)

# Test
#
X = Images[8000:10000, :, :]
D = Labels[8000:10000]

acc = 0
N = len(D)
for k in range(N):
    x = X[k, :, :]

    y1 = Conv_kjw(x, W1)
    y2 = ReLU_kjw(y1)
    y3 = Pool_kjw(y2)
    y4 = numpy.reshape(y3, (-1, 1))
    v5 = numpy.matmul(W5, y4)
    y5 = ReLU_kjw(v5)
    v = numpy.matmul(Wo, y5)
    y = Softmax_kjw(v)

    i = numpy.argmax(y)
    if i == D[k][0]:
        acc = acc + 1

acc = acc / N
print("Accuracy is: ", acc)
################################################################################################################20190408