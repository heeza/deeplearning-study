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
    
    