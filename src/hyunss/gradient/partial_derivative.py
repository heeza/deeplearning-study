import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np 
from gradient.gradient_1d import numerical_diff

# y = x0**2 + x1**2 의 편미분
# x0 = 3, x1 = 4 일때 x0 에 대한 편미분 
def function_tmp(x0):
    return x0**2 + 4.0**2

y = numerical_diff(function_tmp, 3.0) 
print(y)