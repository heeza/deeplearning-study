import numpy as np 
import matplotlib.pyplot as plt 

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01*x**2 + 0.1*x 

def function_2(x):
    return x**2

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d) 
    y = f(x) - d * x 
    
    return lambda t: d * t +  y 

# x = np.arange(0.0, 20.0, 0.1) 
# y = function_2(x)

# tf = tangent_line(function_2, 2) 
# y2 = tf(x)

# plt.xlabel("x")
# plt.ylabel("y")

# plt.plot(x, y)
# plt.plot(x, y2)
# plt.show()