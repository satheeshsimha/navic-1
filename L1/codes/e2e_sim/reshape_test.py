import numpy as np


a = np.array([1,2,3,4,5,6])
print("for -1,2")
b = a.reshape(-1,2)
print(b)
print("sum=",np.sum(b, axis =0))

print("for 2,-1")
b = a.reshape(2,-1).T
print(b)
print("sum=",np.sum(b, axis =0))