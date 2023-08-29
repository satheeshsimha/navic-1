import numpy as np
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.signal
from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm
from numpy_ringbuffer import RingBuffer


class LockIndicator():
    def __init__(self, size, sample_rate, code_length):
      self.buf = RingBuffer(capacity = size, dtype=np.complex_)
      self.sample_rate = sample_rate
      self.code_length = code_length

    def addVal(self,val):
        self.buf.append(val)
    def CN0_cap(self):
        l = len(self.buf)
        if (l == self.buf._capacity):
              sig_power = np.mean(np.abs(np.real(self.buf)))**2
              total_power = np.mean(np.abs(self.buf)**2)
              rho_cap =  (sig_power)/(total_power - sig_power)
              CN0_cap = 10* math.log10(rho_cap) + 10*math.log10(self.sample_rate/2.0) - 10*math.log10(self.code_length)
              return CN0_cap      
        else : 
            return -1
    def phase_lock_indicator(self) :
        l = len(self.buf)
        if (l == self.buf._capacity):
            a = np.sum(np.real(self.buf))**2
            b = np.sum(np.imag(self.buf))**2
            print("a=", a, "b=",b)
            pli =  (a-b)/(a+b)
            return pli  
        else : 
            return -1
        
        
sample_rate = 12 * 1.023*1e6
length = 10230
i = LockIndicator(5,sample_rate , length)

i.addVal(-1+1j)
i.addVal(1+2j)
i.addVal(-3+2j)
print(i.buf)

for k in range(10):
    i.addVal(complex(k+1,k-10))
    print(i.buf)
    print(i.CN0_cap())
    print(i.phase_lock_indicator())