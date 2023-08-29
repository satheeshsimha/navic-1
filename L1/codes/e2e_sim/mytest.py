import numpy as np


indexArr = [0,0,1,1,1,2,2]

mydata = [[1,2,3]]

result = np.array(mydata)[:,indexArr]
#print(result)

a = [0,1,2,3,4]
b = [1,2,2,3,4]

#print (np.sum(np.equal(a,b)))
#print(a[::-1])

c = np.array([[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16],
     [17,18,18,20]])
d = np.empty((1,4), dtype = int)

d = np.append(d, [c[3,:]], axis =0)
d = np.append(d, [c[0,:]], axis =0)
#np.append(d,[c[0,:]])
print(d.shape)
#print(c[:,0])
print(d)