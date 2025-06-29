import numpy as np 
from common.functions import numerical_gradient


def function_2(x):
    return x[0]**2 + x[1]**2 # x[0]의 제곱 + x[1]의 제곱곱
                              
a = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(a)


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad 
    
    return x 

def funcion_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
a = gradient_descent(funcion_2, init_x=init_x, lr=0.1, step_num=100)

print(a)
