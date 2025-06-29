import numpy as np

y = np.array([[0.1, 0.7, 0.2],
              [0.3, 0.2, 0.5],
              [0.8, 0.1, 0.1],
              [0.1, 0.2, 0.3]])

print('전체 배열')
print(y)

print("y[0, 1] = ", y[0, 1])
print("y[1, 2] = ", y[1, 2]) # 1행 2번째 열
print("y[2, 0] = ", y[2, 0]) 

print("y[0][1] = ", y[0][1]) # 비추 느림

print("y[0] = ", y[0]) # 0행 모두
print("y[1] = ", y[1]) # 1행 모두

# 열 전체 가져오기 
print("y[:, 0] = ", y[:, 0]) # 첫번째 열 모두
print("y[:, 1] = ", y[:, 1]) # 두번째 열 모두



print("y.shape = ", y.shape)        # (4, 3)
print("y.shape[0] = ", y.shape[0])  # 4 = 행 개수(첫 번째 차원)
print("y.shape[1] = ", y.shape[1])  # 3 = 열 개수(두 번째 차원)

# 1차원 배열
arr1d = np.array([1, 2, 3, 4, 5])
print("1D shape:", arr1d.shape)         # (5, )
print("1D shape[0]:", arr1d.shape[0])   # 5

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print("2D shape:", arr2d.shape) # (2, 3)
print("2D shape[0]:", arr2d.shape[0]) # 2
print("2D shape[1]:", arr2d.shape[1]) # 3

# 3차원 배열 
arr3d = np.array([
    [[1, 2],[3, 4]],
    [[5, 6],[7, 8]]
])

print("3D shape:", arr3d.shape)

# 고급 인덱싱 예제

y = np.array([[0.1, 0.7, 0.2],
              [0.3, 0.2, 0.5],
              [0.8, 0.1, 0.1]])

rows = [0, 1, 2]
cols = [1, 2, 0]

selected = y[rows, cols]
print("선택된 값들:", selected)

batch_size = 3
t = [1, 2, 0]

correct_probs = y[np.arange(batch_size), t]
print("정답 확률들: ", correct_probs)
