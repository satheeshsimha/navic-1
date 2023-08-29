import numpy as np
from fractions import Fraction

import scipy.constants as sciconst
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.signal
from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm


c = sciconst.speed_of_light
fe = 1575.45e6;              
Dt = 12;                     
DtLin = 10*np.log10(Dt)
Dr = 4;                      
DrLin = 10*np.log10(Dr)
Pt = 44.8;                   
k = sciconst.Boltzmann;  
T = 300;                     
rxBW = 24e6;                 
Nr = k*T*rxBW;

fs = 12 # 50hz
fshift = 0.5 # doppler shift

fsc1 = 1 # Sub carrier1
fsc2 = 6 # Sub carrier 2

epsilon1 = fsc1*1/(100*fs)
epsilon2 = fsc2*1/(100*fs)

pilot_code = [1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,0]
#52276712
sample_pilot_code = np.repeat(pilot_code, fs)

code_length = len(pilot_code)
integtime = code_length

data_code = [1,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1]
#54741411
sample_data_code = np.repeat(data_code, fs)


t = np.arange(code_length*fs)*1/fs
subCarrier1 = np.sign(np.sin(2*np.pi*fsc1*t + epsilon1))
subCarrier2 = np.sign(np.sin(2*np.pi*fsc2*t + epsilon1))

#subCarrier = np.sign(np.sin(2*np.pi*(fsc*t)))

data = [0,0,0,1,1,1,0,1,1,0]
#sample_data = np.repeat(data, fs)

pilot_overlay = [1,1,1,1,1,0,0,0,0,1]
#76041515

k = len(data)

rms = lambda x: np.sqrt(np.mean(np.abs(x)**2, axis=0)) 

#print ("datasig=", datasig, len(datasig))
y_pilot= []
y_data = []

for i in range(4) :
    
    sample_data = np.repeat(data[i], code_length * fs)
    sample_pilot_overlay = np.repeat(pilot_overlay[i], code_length * fs)
    
    
    pilotcode = np.logical_xor(sample_pilot_overlay, sample_pilot_code)
    pilotsig = 1-2*pilotcode
        
    DataSig =1-2*np.logical_xor(sample_data, sample_data_code)
    
    BocPilotSig1 = pilotsig * subCarrier1
    BocDataSig1 = DataSig * subCarrier1
    BocPilotsig6 = pilotsig * subCarrier2
    
    interplexSig = BocPilotSig1* BocDataSig1 *  BocPilotsig6

        
    alpha = (6/11)**0.5
    beta = (4/110)**0.5
    gamma = (4/11)**0.5
    eeta = (6/110)**0.5
    
    iqsig = (alpha*(BocPilotSig1)-beta* (BocPilotsig6 )) + 1j*(gamma*BocDataSig1+eeta*interplexSig)  # Document formula
    
    doppsig = iqsig * np.exp(-1j*2*np.pi*fshift)
    
    #resultsig = np.sum(iqsig)
    waveform = doppsig/rms(doppsig)
    
    rec_pilot_code = (1-2*sample_pilot_code)* subCarrier1 
    rec_data_code = (1-2*sample_data_code) * subCarrier1 
    
    facq = 2 * fshift
    
    iq_sig = waveform * np.exp(-1j*2*np.pi*facq)
    
    iq_p_pilot =  iq_sig * rec_pilot_code
    iq_p_data = iq_sig * rec_data_code
    
    #print(iq_p_pilot.reshape((2, -1)).T)
    
    fllin_pilot = np.sum(iq_p_pilot.reshape((2, -1)).T, axis=0)
    print("i=",i)
    #print("fllin pilot=", fllin_pilot)
    
    fllin_data = np.sum(iq_p_data.reshape((2, -1)).T, axis=0)
    print("fllin data=", fllin_data)
    
    phasor_pilot = np.conj(fllin_pilot[0])*fllin_pilot[1]
    fqyerr_pilot = -1*np.angle(phasor_pilot)/(np.pi*integtime)
    
    phasor_data = np.conj(fllin_data[0])*fllin_data[1]
    fqyerr_data= -1*np.angle(phasor_data)/(np.pi*integtime)
    
    print("Freq error pilot=",fqyerr_pilot)
    #print("Freq error data=",fqyerr_data)

    y_pilot = np.append(y_pilot, np.mean(waveform * rec_pilot_code))
    y_data = np.append(y_data, np.mean(waveform * rec_data_code))


mapbits = lambda l: np.piecewise(l, [l < 0, l >= 0], [1, 0])

print("recd_data_bits =", mapbits(np.imag(y_data)))
print("recd_pilot_bits =", mapbits(np.real(y_pilot)))
#print("recd_data_bits =", y_data)
#print("recd_pilot_bits =", y_pilot)