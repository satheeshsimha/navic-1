import numpy as np

a= []
for i in range(20):
    a.append(i)
for k in range (5) :
    a.append(i-k-1)

l = a[-1]

for j in range(l):
    a.append(l+j+1)

l = a[-1]

for j in range(l):
    a.append(l-j-1)
    
l = a[-1]

for j in range(l-5):
    a.append(l+j+1)

l = a[-1]

for j in range(l):
    a.append(l-j-1)

l = a[-1]

for j in range(l-5):
    a.append(l+j+1)

l = a[-1]

for j in range(l):
    a.append(l-j-1)

final = np.array(a)

print(final)

i = len(final)


max = 0
for j in range(i) :
   print(j,final[j])
   if (max > final[j] and final[j] == 20):
       print ("my max is ", j, final[j])
   else :
       max = final[j]
   

