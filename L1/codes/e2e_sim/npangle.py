import numpy as np

a = -0.00258 + 0.02179j
b = 0.0470 - 0.0199j
c = -0.00975 + 0.0162j
#d = -0.08845 + 0.0471j
d= 0.00408 + 0.0072j

comp_val = np.array([-0.00258 + 0.02179j, 0.0470 - 0.0199j, -0.00975 + 0.0162j, 0.00408 + 0.0072j ])

k = len(comp_val)
integtime = 0.01

for i in range(k) :
    ang = np.angle(comp_val[i])
    if (ang < 0) :
        print(-ang/(np.pi*integtime))
    else :
        print((2*np.pi - ang)/(np.pi*integtime))



#-1*np.angle(-0.00975+0.0162j)/(np.pi*integtime)

print("a angle = ", np.angle(a)*360/(2*np.pi))
print("b angle = ", np.angle(b)*360/(2*np.pi))
print("c angle = ", np.angle(c)*360/(2*np.pi))
print("d angle = ", np.angle(d)*360/(2*np.pi))
print("\n")

print("a fqyerr = ", -1*np.angle(a)/(np.pi*integtime))
print("b fqyerr = ", -1*np.angle(b)/(np.pi*integtime))
print("c fqyerr = ", -1*np.angle(c)/(np.pi*integtime))
print("d fqyerr = ", -1*np.angle(d)/(np.pi*integtime))
print("\n")

print("a fqyerr = ", (2*np.pi-1*np.angle(a))/(np.pi*integtime))
print("b fqyerr = ", (2*np.pi-1*np.angle(b))/(np.pi*integtime))
print("c fqyerr = ", (2*np.pi-1*np.angle(c))/(np.pi*integtime))
print("d fqyerr = ", (2*np.pi-1*np.angle(d))/(np.pi*integtime))
print("\n")


#a1 = 0.1025 + 0.1119j
#a2 = -0.1173 + 0.0844j
#dot = np.real(a1) * np.real(a2)+ np.imag(a1) * np.imag(a2)
#cross = np.real(a1) * np.imag(a2) - np.real(a2) * np.imag(a1)

#print("dot=", dot, "cross=", cross)

#print((np.arctan2(np.real(a),np.imag(a))*(180/np.pi)/(integtime)))
#print((np.arctan2(np.real(b),np.imag(b))*(180/np.pi)/(integtime)))
#print((np.arctan2(np.real(c),np.imag(c))*(180/np.pi)/(integtime)))
#print((np.arctan2(np.real(d),np.imag(d))*(180/np.pi)/(integtime)))

print(np.arctan2(np.imag(a), np.real(a)))
print(np.angle(a))

