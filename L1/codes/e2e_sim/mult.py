import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([-1,-1,-1,-1,-1])
c = a.reshape(-1,1)
d = b.reshape(-1,1)
print(c.shape)
print(d.shape)
print(a*b)
print(c*d)