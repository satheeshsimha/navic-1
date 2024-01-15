import numpy as np
import csv
import struct
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import scipy.signal
from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm
from numpy_ringbuffer import RingBuffer
from random import *
from ldpc import bp_decoder
import l1functions as l1

dataCodeLength = 10230
pilotCodeLength = 10230
pilotOverlayCodeLength = 1800
codeFreqBasis = 1.023e6

bufsize_power_estimation = 10  # number of samples to be considred for estimating SNR
cn0_min = 25 # Min  Squared Signal-to-Noise Variance estimator in db-Hz
detector_threshold = 0.85 # Phase lock detector threshold
lock_fail_counter_threshold = 25 #Maximum value for the lock fail counter
lock_counter_threshold = 20



sampleRate = 4e6
samplePeriod = 1/sampleRate
symbolRate = 100
satId = np.arange(1,65)
#satId = np.array([2,3,4,6])
numChannel = len(satId)

#frequrency shift to be applied to the signal
fShift = np.array([489, 1299, 3796, 4888])
channelpfo = l1.PhaseFrequencyOffset(sampleRate)
sigDelay = np.array([300.34, 587.21, 488.32, 531.78])
dynamicDelayRange = 50
staticDelay = np.round(sigDelay - dynamicDelayRange)
channelstatd = l1.IntegerDelay(staticDelay)
channelvard = l1.FractionalDelay(1, 65535)

PLLIntegrationTime = 10e-3
PLLNoiseBandwidth = 18 # In Hz
FLLNoiseBandwidth = 2 # In Hz
DLLNoiseBandwidth = 1  # In Hz


#simulation duration, steps at which values are recorded(here for every 10ms)
simDuration = 36

timeStep = PLLIntegrationTime
numSteps = round(simDuration/timeStep)
samplePerStep = int(timeStep/samplePeriod)


codeTable_data = l1.genNavicCaTable_Data(sampleRate, dataCodeLength,codeFreqBasis, satId)
codeTableSampCnt_data = len(codeTable_data)
#print("code table data ",codeTable_data.shape)

codeTable_pilot = l1.genNavicCaTable_Pilot(sampleRate, pilotCodeLength,codeFreqBasis,  satId)
codeTableSampCnt_pilot = len(codeTable_pilot)
#print("pilot count ",codeTableSampCnt_pilot)


c = sciconst.speed_of_light
fe = 1575.42e6;              
Dt = 12;                     
DtLin = 10*np.log10(Dt)
Dr = 4;                      
DrLin = 10*np.log10(Dr)
Pt = 44.8;                   
k = sciconst.Boltzmann;  
T = 300;                     
rxBW = 24e6;                 
Nr = k*T*rxBW;              




sqrtPr = np.sqrt(Pt*DtLin*DrLin)*(1/(4*np.pi*(fe+fShift)*sigDelay*samplePeriod))



rms = lambda x: np.sqrt(np.mean(np.abs(x)**2, axis=0)) 

datagen = l1.NavicDataGen(symbolRate, sampleRate, numChannel)
pilotOverlayCodegen = l1.PilotOverlayBitGen(satId, pilotOverlayCodeLength, symbolRate, sampleRate)
modulator = l1.NavicL1sModulator(sampleRate)
istep = 0
waveform = []
for istep in range(numSteps):
    
    #Navigation Data
    navdata = datagen.GenerateBits(timeStep)
    
    #Pilot Overlay Code
    pilotOverlayCode = pilotOverlayCodegen.GenerateBits(timeStep)


    #Baseband modulation
    iqsig = modulator.Modulate(navdata,codeTable_data[:,np.arange(numChannel)],codeTable_pilot[:,np.arange(numChannel)],pilotOverlayCode)

    # Doppler shift
    doppsig = channelpfo.Offset(iqsig, fShift)

    # Delay
    staticDelayedSignal = channelstatd.Delay(doppsig)
    leftoutDelay = sigDelay - staticDelay
    delayedSig = channelvard.Delay(staticDelayedSignal, leftoutDelay)

    # Power scaling
    scaledSig = l1.PowerScale(delayedSig, sqrtPr)
    
    # Add signals from each channel
    resultsig = np.sum(scaledSig, axis=1)
    # Generate noise
    noisesig = (np.random.normal(scale=Nr**0.5, size=(samplePerStep, )) + 1j*np.random.normal(scale=Nr**0.5, size=(samplePerStep, )))/2**0.5

    finalsig = resultsig+noisesig
    waveform = np.append(waveform, finalsig)
    # Add thermal noise to composite signal
    rxwaveform = resultsig

"""## Bit and Frame Synchronization"""

pilot_overlay_sent = pilotOverlayCodegen.GetBitStream()
navbits = datagen.GetSymbolStream() #Encrypted symbols
navic_raw_data = datagen.GetDataStream()
pilot_overlay_sent = pilot_overlay_sent.astype(np.int16)
navbits = navbits.astype(np.int16)

waveform = waveform / rms(waveform)
print(np.shape(waveform))



print("TRANSMITTED WAVEFORM:",waveform)

np.savetxt("fc70M_fs4M_encryptedsymbols_wo_fractional_delay.txt",navbits)

np.savetxt("fc70M_fs4M_pilot_overlay_sent_wo_fractional_delay.txt",pilot_overlay_sent)

np.savetxt("fc70M_fs4M_rawdata_wo_fractional_delay.txt",navic_raw_data)


samp_i = np.real(waveform)*2048.0
samp_i_len = len(samp_i)
#print(samp_i)
for a in range(samp_i_len):
    samp_i[a] = round(samp_i[a])
    if samp_i[a]>2047.0:
        samp_i[a]=2047
    elif samp_i[a]<-2048.0:
        samp_i[a]=-2048
samp_q = np.imag(waveform)*2048.0
samp_q_len = len(samp_q)
#print(samp_q)
for b in range(samp_q_len):
    samp_q[b] = round(samp_q[b])
    if samp_q[b]>2047.0:
        samp_q[b]=2047
    elif samp_q[b]<-2048.0:
        samp_q[b]=-2048
#print(samp_i)
#print(samp_q)
sig = np.stack((samp_i,samp_q),axis=1)
np.savetxt("fc70M_fs4M_wo_fractional_delay.csv", sig,delimiter=",")
l1.csv2bin( "fc70M_fs4M_wo_fractional_delay.csv", "fc70M_fs4M_wo_fractional_delay.sc16q11")



