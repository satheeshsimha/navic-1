import numpy as np
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import scipy.signal as signal

subframe1_navIn = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0])
subframe1_navOut = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0])

subframe2_navIn = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
subframe2_navOut = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])

subframe3_navIn = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1])
subframe3_navOut = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1])
fnt_size = 8

xaxis1 = np.arange(0,9)

plt.subplot(6,1,1)
plt.step(xaxis1,subframe1_navIn)

#plt.plot(subframe1_navIn)
    #plt.ylim([0,0.05])

plt.ylabel('SF1 I/p',fontsize= fnt_size)
    
plt.subplot(6,1,2)
plt.step(xaxis1,subframe1_navOut)
#plt.plot(subframe1_navOut)
    #plt.ylim([0,0.05])
#plt.xlabel('time') ; 
plt.ylabel('SF1 O/p',fontsize= fnt_size)

xaxis1 = np.arange(0,24)

plt.subplot(6,1,3)
plt.step(xaxis1,subframe2_navIn)
#plt.plot(subframe2_navIn)
    #plt.ylim([0,0.05])
#plt.xlabel('time') ; 
plt.ylabel('SF2 I/p',fontsize= fnt_size)

plt.subplot(6,1,4)
plt.step(xaxis1,subframe2_navOut)
#plt.plot(subframe2_navOut)
    #plt.ylim([0,0.05])
#plt.xlabel('time') ; 
plt.ylabel('SF2 O/p',fontsize= fnt_size)

plt.subplot(6,1,5)
plt.step(xaxis1,subframe3_navIn)
#plt.plot(subframe3_navIn)
    #plt.ylim([0,0.05])
#plt.xlabel('time') ; 
plt.ylabel('SF3 I/p',fontsize= fnt_size)
    
plt.subplot(6,1,6)
plt.step(xaxis1,subframe3_navOut)
#plt.plot(subframe3_navOut)
    #plt.ylim([0,0.05])
#plt.xlabel('time') ; 
plt.ylabel('SF3 O/p',fontsize= fnt_size)
plt.xlabel('Nav bit number') ;   
    
plt.savefig('./mynavbits.png',dpi=1200)
plt.show()
