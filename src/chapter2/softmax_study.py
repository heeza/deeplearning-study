import numpy as np 
from common.functions import softmax

a = np.array([1010, 1000, 990])
# np.exp(a) / np.sum(np.exp(a)) 

c = np.max(a)
print(c)

d = a - c
print(d)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(np.sum(y))
