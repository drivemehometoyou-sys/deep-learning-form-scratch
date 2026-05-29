import numpy as np

## 多维数组的建立
# A = np.array([1,2,3,4])
# print(A)
# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])

# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)


## 矩阵乘法
# A = np.array([[1,2],[3,4]])
# B = np.array([[5,6],[7,8]])
# print(np.dot(A,B))

# A = np.array([[1,2,3],[4,5,6]])
# print(A)
# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.dot(A,B))



X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
print(X)
print(W)
Y = np.dot(X,W)
print(Y)