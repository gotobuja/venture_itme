## 20190319_v1_0_0, 최초 생성
## 20190401_v1_0_1, Accuracy가 원본 대비 0.02 ~ 0.03 낮게 나옴, 'return y' 명령어를 for문 밖에 있도록 수정

import numpy
from scipy import signal

def Conv_kjw(x, W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol) = x.shape

    ## 컨벌루션 연산 후 나오는 행렬 크기
    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1

    y = numpy.zeros((yrow, ycol, numFilters))

    for k in range(numFilters):
        filter = W[:, :, k]
        ## squeeze: 각 차원에서 1차원에 해당 하는 것을 0차원으로 변환
        ## rot90: 반시계 방향으로 행렬을 돌림, 여기서는 2번 회전
        filter = numpy.rot90(numpy.squeeze(filter), 2)
        y[:, :, k] = signal.convolve2d(x, filter, 'valid')

    return y