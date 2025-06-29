import numpy as np 
from sympy import Symbol, solve
import matplotlib.pyplot as plt

a = np.array([2, 2])
b = np.array([2, -1])

print(a + b)
print(a - b)

print(2 * a)

a = Symbol('a')
b = Symbol('b')

ex1 = -1 * a + b - 2
ex2 = 2 * a + b - 4 

print(solve((ex1, ex2)))

k = Symbol('k')
t = Symbol('t')

ex1 = 4 * k - 4 * t 
ex2 = -6 * k - 2 * t + 4 
print(solve((ex1, ex2)))

import math 

print(10 * math.cos(math.radians(60)))

# 기여도 = 이동거리 * 힘의크기
# 기여도 = 3m 이동 * (10(힘) * cos60도 (각도)) 
# 기여도 = 3 * math.cos(math.radius(60)) = 15
# 이건 곧 벡터의 내적이다
# a . b = |a||b|cos세타
# |a||b|cos = a1b1 + a2b2
