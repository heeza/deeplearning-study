import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pylab as plt 
from common.functions import numerical_diff

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1) # 0 ~ 20까지, 0.1씩 뛰면서 (20은 미포함)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x) 

plt.plot(x, y)
plt.plot(x, y2)
plt.show()