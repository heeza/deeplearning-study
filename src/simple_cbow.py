import sys
sys.path.append('..')
import numpy as np 
from common.layers import MatMul, SoftmaxWithLoss 

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

    