## 20190319_v1_0_0, 최초 생성

import numpy
from scipy import signal

def Pool_kjw(x):
    (xrow, xcol, numFilters) = x.shape

    ## 풀링 후의 행렬 크기
    y = numpy.zeros((int(xrow/2), int(xcol/2), numFilters))

    for k in range(numFilters):
        filter = numpy.ones((2, 2))/(2*2)
        image = signal.convolve2d(x[:, :, k], filter, 'valid')

        ## 첫 번째 값 부터 한 칸씩 띄어서 값 가져오기
        y[:, :, k] = image[::2, ::2]

    return y