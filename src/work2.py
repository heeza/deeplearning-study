import numpy as np
from common.layers import MatMul

c = np.array([1, 0, 0, 0, 0, 0, 0])
W = np.random.randn(7, 3)
# h = np.matmul(c, W)
# print(h)

layer = MatMul(W)
a = layer.forward(c)
print(a) 
