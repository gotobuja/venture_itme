## 20190319_v1_0_0, 최초 생성

"""
    Originally written by Martin Thoma
        http://martin=thoma.com/classify-mnist-with-pybrain/
"""

from struct import unpack
import gzip
from numpy import uint8, zeros, float32

# Read input images and labels(0-9).
# Return it as list of tuples.
#

# example parameter
imagefile = 'MNIST\\t10k-images-idx3-ubyte.gz'
labelfile = 'MNIST\\t10k-labels-idx1-ubyte.gz'

def LoadMnistData_kjw(imagefile, labelfile):
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data
    # We have to get big ending unsigned int. So we need '>|' ## register에서 memory로 데이터가 이동하는 방식을 이야기 하는것 같다.

    # Get metadata for images
    images.read(4) # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>l', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>l', rows)[0]
    cols = images.read(4)
    cols = unpack('>l', cols)[0]

    # Get metadata for labels
    labels.read(4) # skip the magic_number
    N = labels.read(4)
    N = unpack('>l', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32) # Initialize numpy array
    y = zeros((N, 1), dtype=uint8) # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)

        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1) # Just a signal byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel

        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    return (x, y)