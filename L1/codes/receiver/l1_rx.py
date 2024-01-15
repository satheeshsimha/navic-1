import numpy as np
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import scipy.signal
import scipy.signal as signal
from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm
from numpy_ringbuffer import RingBuffer
from random import *
from ldpc import bp_decoder
import l1functions_mod as l1
 
dataCodeLength = 10230
pilotCodeLength = 10230
codeFreqBasis = 1.023e6

bufsize_power_estimation = 10  # number of samples to be considred for estimating SNR
cn0_min = 25 # Min  Squared Signal-to-Noise Variance estimator in db-Hz
detector_threshold = 0.85 # Phase lock detector threshold
lock_fail_counter_threshold = 25 #Maximum value for the lock fail counter
lock_counter_threshold = 20



#sampleRate = 2048000
sampleRate = 4000000
samplePeriod = 1/sampleRate
#satId = np.array([10])
satId = np.arange(1,65)
numChannel = len(satId)

#sigDelay = np.array([300.34, 587.21, 488.32, 531.78])
#dynamicDelayRange = 50
#staticDelay = np.round(sigDelay - dynamicDelayRange)

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
#codeTable_data = l1.genNavicCaTable_Data(sampleRate, dataCodeLength,codeFreqBasis, np.arange(1,65))
codeTableSampCnt_data = len(codeTable_data)

codeTable_pilot = l1.genNavicCaTable_Pilot(sampleRate, pilotCodeLength,codeFreqBasis,  satId)
#codeTable_pilot = l1.genNavicCaTable_Pilot(sampleRate, pilotCodeLength,codeFreqBasis,  np.arange(1,65))
codeTableSampCnt_pilot = len(codeTable_pilot)


i=0
waveform = []
istep = 0 

rx_iq_arr = []

rx_iq_arr=l1.bin2csv("rx_samples_fcl1_fs4M_wireless_gain10_fb10M.sc16q11", "rxwave.csv",chunk_bytes=4096)

satVis = 0
for istep in range(numSteps):
    
    #waveform = rx_iq_arr[istep*samplePerStep:(istep+5)*samplePerStep] 
    waveform = rx_iq_arr[(istep+500)*samplePerStep:(istep+(500+10))*samplePerStep] 
    
    # Perform acquisition once from cold-start
    if satVis==0:

        # Acqusition doppler search space
            fMin = -5000
            fMax = 5000
            #The FLL discriminator has a bandwidth of +- 50 , for 10 ms integration duration
            #Hence, to avoid border cases of +- 50, fStep is taken as 80 so that we can have
            # an error of +- 40
            fStep = 80
            fSearch = np.arange(fMin, fMax + fStep , fStep)

            tracker = []
            sat = []
            
            print("istep:",istep)
            
        # Perform acquisition for each satellite
            for prnId in range(numChannel):
                status, codePhase, doppler = l1.navic_pcps_acquisition(
                                                                waveform, 
                                            codeTable_pilot[np.arange(0, samplePerStep)%codeTableSampCnt_pilot, prnId],
                                            sampleRate ,fSearch,10 ,1
                                        )   
                delaySamp = codePhase
                codePhase = (codePhase % codeTableSampCnt_data)/(sampleRate/codeFreqBasis)
                
                print("PRN ID: ", satId[prnId], "processed") 
                
                # If a satellite is visible, initialize tracking loop
                if(status == True):
                    
                    print(f"Acquisition results  for PRN ID {satId[prnId]}\n Status:{status} Doppler:{doppler} Delay/Code-Phase:{delaySamp}/{codePhase}\n")
                    satVis += 1
                    sat= np.append(sat,satId[prnId])
                    tracker.append(l1.NavicTracker(satId[prnId]))
                    tracker[-1].SampleRate = sampleRate
                    tracker[-1].CenterFrequency = 0
                    tracker[-1].PLLNoiseBandwidth = PLLNoiseBandwidth
                    tracker[-1].FLLNoiseBandwidth = FLLNoiseBandwidth
                    tracker[-1].DLLNoiseBandwidth = DLLNoiseBandwidth
                    tracker[-1].PLLIntegrationTime = round(PLLIntegrationTime*1e3)
                    tracker[-1].PRNID = satId[prnId]
                    tracker[-1].InitialDopplerShift = doppler
                    tracker[-1].delaySamp = delaySamp
                    tracker[-1].setupImpl(dataCodeLength, bufsize_power_estimation, cn0_min, detector_threshold, lock_fail_counter_threshold)
                    tracker[-1].resetImpl()

            trackDataShape = (round(numSteps), satVis)
            y_pilot = np.zeros(trackDataShape, dtype=np.complex_)
            y_data = np.zeros(trackDataShape, dtype=np.complex_)
            fqyerr = np.zeros(trackDataShape)
            fqynco = np.zeros(trackDataShape)
            pherr = np.zeros(trackDataShape)
            phnco = np.zeros(trackDataShape)
            delayerr = np.zeros(trackDataShape)
            delaynco = np.zeros(trackDataShape)
            cn0_cap = np.zeros(trackDataShape)
            pli = np.zeros(trackDataShape)
            lock_fail_counter = np.zeros(trackDataShape)
            fc = np.zeros(trackDataShape)

	

    # Perform tracking for visible satellite
    for i in range(satVis):
        waveform = rx_iq_arr[istep*samplePerStep+tracker[i].delaySamp:(istep+1)*samplePerStep+tracker[i].delaySamp] 
        y_pilot[istep, i], y_data[istep,i],fqyerr[istep, i], fqynco[istep, i], pherr[istep, i], phnco[istep, i], delayerr[istep, i], delaynco[istep, i], cn0_cap[istep,i], pli[istep,i], lock_fail_counter [istep,i],fc[istep,i] = tracker[i].stepImpl(waveform)

k = 1800 
print(k)
if len(sat)==0:
    print("No satellites were detected.")
else:
    sat=sat.astype(np.int16)
    print(sat)

print("Visible satellites\n",satVis)
for i in range(satVis):
    navic_raw_data = np.loadtxt("fcl1_fs4M_rawdata.txt", dtype=float)
    navic_raw_data = navic_raw_data.astype(int)
    navic_raw_data=navic_raw_data[:,i]

    navbits = np.loadtxt("fcl1_fs4M_encryptedsymbols.txt", dtype=float)
    navbits = navbits.astype(int)
    navbits = navbits[:,i]
    
    pilot_overlay_sent = np.loadtxt("fcl1_fs4M_pilot_overlay_sent.txt", dtype=float)
    pilot_overlay_sent = pilot_overlay_sent.astype(int) 
    pilot_overlay_sent = pilot_overlay_sent[:,i]
    pilot_overlay_sent = pilot_overlay_sent[:1800]
    
    pilot_overlay_local_gen = l1.genNavicCaCode_Pilot_Overlay(sat[i])
    pilot_overlay_local_gen = pilot_overlay_local_gen.astype(np.int16)
    #print(pilot_overlay_local_gen==pilot_overlay_sent)
    
    # Find the index for which the lock counter is 20, after reaching a max lock counter value 
    # or index of max counter value less than 20
    max = 0
    maxval_lock_counter_idx = np.argmax(lock_fail_counter[:,i])
    maxval_lock_counter = lock_fail_counter[maxval_lock_counter_idx,i]
    print ("Maximum locked index is ", maxval_lock_counter_idx, maxval_lock_counter)
    if (maxval_lock_counter <= lock_counter_threshold):
        locked_sample_index = maxval_lock_counter_idx
        print ("Outer if: my locked index is ", maxval_lock_counter_idx, maxval_lock_counter)
        lckdidx = maxval_lock_counter_idx
    else :
        for idx in range(k) :
            if (max > lock_fail_counter [idx,i] and lock_fail_counter [idx,i] == lock_counter_threshold):
                locked_sample_index = idx
                print ("Inner if my locked index is ", idx, lock_fail_counter [idx,i])
                lckdidx = idx
                break
            else :
                max = lock_fail_counter [idx,i]
    
    #print(lckdidx)
    n = 1 #Number of data per bit
    skip = 0 #Forgo few bits as the tracking loops starts early
    databits_received = np.real(y_data[n*skip:,i])
    pilotOverlay_received = np.imag(y_pilot[n*skip:,i])
    
    mapbits = lambda l: np.piecewise(l, [l < 0, l >= 0], [1, 0])
    mapbits_inverted = lambda l: np.piecewise(l, [l < 0, l >= 0], [0, 1])
    bits = mapbits(databits_received)
    overlay = mapbits(pilotOverlay_received)
    overlay = overlay.astype(int)
    corr_output = signal.correlate(overlay,pilot_overlay_local_gen,mode='valid')
    max_corr_out = (np.argmax(corr_output))

    bits_inverted = mapbits_inverted(databits_received)
    overlay_inverted = mapbits_inverted(pilotOverlay_received)
    overlay_inverted = overlay_inverted.astype(int)
    corr_inverted_output = signal.correlate(overlay_inverted,pilot_overlay_local_gen,mode='valid')
    max_corr_inverted_out = (np.argmax(corr_inverted_output))

    if (corr_output[max_corr_out]>=corr_inverted_output[max_corr_inverted_out]):
        overlay = overlay[max_corr_out:1800+max_corr_out]
        bits = bits_inverted[max_corr_out:1800+max_corr_out]
    else:
        overlay = overlay_inverted[max_corr_inverted_out:1800+max_corr_inverted_out]
        bits = bits[max_corr_inverted_out:1800+max_corr_inverted_out]

    if (max_corr_out==1800 or max_corr_inverted_out==1800):
        navbits = navbits[1800:]
        navic_raw_data = navic_raw_data[883:1766]
    else:
        navbits = navbits[:1800]
        navic_raw_data = navic_raw_data[:883]
    
    for iter in range(k) :
            #print("iter=", iter, "data sent=",navbits[iter,i], "data recd=",bits[iter], navbits[iter,i]==bits[iter], pherr[iter,i],phnco[iter,i], fc[iter,i], lock_fail_counter[iter,i]  )
            print("iter=", iter, "data sent=",navbits[iter], "data recd=",bits[iter], navbits[iter]==bits[iter], "overlay sent=", pilot_overlay_local_gen[iter], "overlay recd=",overlay[iter], pilot_overlay_local_gen[iter]==overlay[iter], fqyerr[iter,i], lock_fail_counter[iter,i])
    

    decode=l1.decoder()
    
    #print("Bits shape is:",bits.shape)

    np.set_printoptions(threshold=np.inf)
    s1_decoded,s2_decoded,s3_decoded=decode.subframes_decode(bits)
    
    print("Subframe1 Before Encoding (9 bits):",navic_raw_data[:9])
    print("Subframe1 After Encoding (52 Symbols):",navbits[:52])
    print("Subframe1 After Receiving (52 Symbols):",bits[:52])
    print("Subframe1 After decoding (9 bits):",s1_decoded)
    print("Subframe 1 is same before encoding and after decoding:",np.all(navic_raw_data[:9]==s1_decoded))

    print("Subframe2 Before Encoding (600 bits):",navic_raw_data[9:609])
    print("Subframe2 After Encoding (1200 Symbols):",navbits[52:1252])
    print("Subframe2 After Receiving (1200 Symbols):",bits[52:1252])
    print("Subframe2 After decoding (600 bits):",s2_decoded)
    print("Subframe 2 is same before encoding and after decoding:",np.all(navic_raw_data[9:609]==s2_decoded))
        
    #test code for crc

    subframe2_decoded_string = ''.join(str(x) for x in s2_decoded)
    subframe2_decoded_int = int(subframe2_decoded_string,2)
    subframe2_decoded_hex= hex(subframe2_decoded_int)
    sub = ''.join(str(e) for e in subframe2_decoded_hex)
    subframe2decodedint = int(sub,16)
        
    decode.printlongdiv(subframe2decodedint, 25578747)
 
    print("Subframe3 Before Encoding (274 bits):",navic_raw_data[609:])
    print("Subframe3 After Encoding (548 Symbols):",navbits[1252:])
    print("Subframe3 After Receiving (548 Symbols):",bits[1252:])
    print("Subframe3 After decoding (274 bits):",s3_decoded)
    print("Subframe 3 is same before encoding and after decoding:",np.all(navic_raw_data[609:]==s3_decoded))

    #test code for crc

    subframe3_decoded_string = ''.join(str(x) for x in s3_decoded)
    subframe3_decoded_int = int(subframe3_decoded_string,2)
    subframe3_decoded_hex= hex(subframe3_decoded_int)
    sub3 = ''.join(str(e) for e in subframe3_decoded_hex)
    subframe3decodedint = int(sub3,16)
        
    decode.printlongdiv(subframe3decodedint, 25578747)
 

    print("k=", k)
    
    plt.subplot(6,1,1)
    plt.plot(fqyerr[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Fqy Error')
    
    plt.subplot(6,1,2)
    plt.plot(fqynco[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Fqy NCO')

    plt.subplot(6,1,3)
    plt.plot(pherr[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Phase Error')

    plt.subplot(6,1,4)
    plt.plot(phnco[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Phase NCO')

    plt.subplot(6,1,5)
    plt.plot(delayerr[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Delay Error')
    
    plt.subplot(6,1,6)
    plt.plot(delaynco[:,i])
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Delay NCO')
    
    
    plt.savefig('./myplot.png')
    plt.show()

    
