import sys, os
import numpy as np 
from dataset.mnist import load_mnist
from PIL import Image 

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(X_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = X_train[0]
label = t_train[0]
print(label)

print(img.shape) 
img = img.reshape(28, 28)
print(img.shape)

img_show(img)