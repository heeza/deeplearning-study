class MulLayer:
    def __init__(self):
        self.x = None 
        self.y = None 

    def forward(self, x, y):
        self.x = x 
        self.y = y 
        out = x * y 

        return out
    
    def backward(self, dout):
        dx = dout * self.y # x와 y를 바꾼다
        dy = dout * self.x 

        return dx, dy 
    

apple = 100
apple_num = 2
tax = 1.1 

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax) 

print(price)

# 역전파 
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax) 
