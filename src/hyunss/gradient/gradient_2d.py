import numpy as np 

def numerical_gradient(f, X):
    h = 1e-4
    grad = np.zeros_like(X) 

    for idx in range(X.size):
        tmp_val = X[idx]

        X[idx] = tmp_val + h 
        fxh1 = f(X) 

        X[idx] = tmp_val - h 
        fxh2 = f(X) 

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        X[idx] = tmp_val

    return grad         

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x 

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad 

    return x

# 사용 샘플

def function_2(x):
    return x[0]**2 + x[1]**2

y = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(y)

init_x = np.array([-3.0, 4.0])
y = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
print(y)



