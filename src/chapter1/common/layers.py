from common.np import *
from common.config import GPU
from common.functions import softmax, cross_entropy_error

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None 

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out 
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)      # 출력의 변화가 어떻게 전달되는지 확인
        dW = np.dot(self.x.T, dout) # 가중치의 변화가 어떻게 전달되는지 확인
        self.grads[0][...] = dW 
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None 
        self.t = None 

    def forword(self, x, t):
        self.t = t 
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backword(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout 
        dx = dx / batch_size 

        return dx
    
class Relu:
    def __init__(self):
        self.mask = None 
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None 
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 

        return out 
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out 
        
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W 
        self.b = b 
        self.x = None 
        self.dW = None 
        self.db = None 

    def forward(self, x):
        self.x = x 
        out = np.dot(x, self.W) + self.b 

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout) 
        self.db = np.sum(dout, axis=0) 

        return dx 
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None # softmax 출력
        self.t = None # 정답 레블 

    def forward(self, x, t):
        self.t = t 
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t) 
        return self.loss 
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 

        return dx 
    
    