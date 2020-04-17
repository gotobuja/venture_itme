## 20190319_v1_0_0, 최초 생성

import numpy

def Softmax_kjw(x):
    ## subtract: x에서 numpy.max(x)를 뺀다, overflow 방지.
    x = numpy.subtract(x, numpy.max(x))

    ## 자연상수(e)의 지수 함수
    ex = numpy.exp(x)

    return ex / numpy.sum(ex)