import numpy as np
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.signal import hilbert


from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm
from numpy_ringbuffer import RingBuffer
from random import *
from ldpc import bp_decoder

import l1_boc11 as l1
import navic_l1_navbits_decoder as navdata

SPEED_OF_LIGHT  = 299792458.0

def read_from_file(file,samples_in_10ms,bytes_per_sample):
    
    data = file.read(samples_in_10ms*bytes_per_sample)
    # Convert the binary data to integers (int16 for I and Q)
    if bytes_per_sample == 2:
        samples = np.frombuffer(data, dtype=np.int8)
    elif bytes_per_sample == 4:
        samples = np.frombuffer(data, dtype=np.int16)

    # Separate I and Q (assumes interleaved I/Q format)
    I_samples = samples[0::2]  # Take every even index for I
    Q_samples = samples[1::2]  # Take every odd index for Q

    # Combine I and Q as complex samples (if needed)
    complex_samples = I_samples + 1j * Q_samples
    
    return complex_samples


 

#main program
#code chip rate, sample rate and sample period
#refer to navicsim.py for all function detials
dataCodeLength = 10230
pilotCodeLength = 10230
pilotOverlayCodeLength = 1800
codeFreqBasis = 1.023e6

bufsize_power_estimation = 10  # number of samples to be considred for estimating SNR
cn0_min = 25 # Min  Squared Signal-to-Noise Variance estimator in db-Hz ; 20
detector_threshold = 0.85 # Phase lock detector threshold ; 0.85
lock_fail_counter_threshold = 25 #Maximum value for the lock fail counter
lock_counter_threshold = 20


sampleRate = 4.096e6
#sampleRate = 2.048e6
#sampleRate = 18.548387e6
#sampleRate = 10*codeFreqBasis
samplePeriod = 1/sampleRate
symbolRate = 100
symbol_Period_inms = 1000/symbolRate


PLLIntegrationTime = 10e-3
PLLNoiseBandwidth = 40 # In Hz
#PLLNoiseBandwidth = 18 # In Hz
#FLLNoiseBandwidth = 4 # In Hz
FLLNoiseBandwidth = 2 # In Hz
DLLNoiseBandwidth = 1 # In Hz


#simulation duration, steps at which values are recorded(here for every 10ms)
simDuration = 36


timeStep = PLLIntegrationTime

numSteps = round(simDuration/timeStep)
samplePerStep = round(timeStep/samplePeriod)
samples_in_10ms = round(symbol_Period_inms*1e-3/samplePeriod)
numSymbols = simDuration * symbolRate
numIntegrations_in10ms = int(symbol_Period_inms/(PLLIntegrationTime*1e3))
bytes_per_sample = 4
no_of_correlators = 4 # Number of correlators in tracking loop other than Prompt

satVis = 0
istep = 0 
coh_int =1
non_coh_int = 20


if_frequency = 0

waveform=[]
satVis = 0

file = open('/home/satheeshsk/navic/myL1/signalsFromSAtellite/navic_bin_files/navic_l1_samples/navic_bin_files/rx_samples_navic_l1_4.096Mhz_from_ee_block_terrace2.bin',"rb")

#read 20 msec from file
waveform = read_from_file(file, samples_in_10ms,bytes_per_sample)
waveform = np.append(waveform,read_from_file(file, samples_in_10ms,bytes_per_sample))

tracker = []
nav_data = []
sat = []

incr_tow = 0
for istep in range(numSteps):
    
    if satVis == 0:

        # Acqusition doppler search space
            fMin = -5000
            fMax = 5000
            fStep = 50
            fSearch = np.arange(fMin, fMax + fStep , fStep)

            tracker = []
            sat = []
            
        # Perform acquisition for each satellite
            for prnId in range(10,11,1):
                
                codeTable_data = l1.genNavicCaTable_Data(sampleRate, dataCodeLength,codeFreqBasis, np.array([prnId]))
                codeTableSampCnt_data = len(codeTable_data)


                codeTable_pilot = l1.genNavicCaTable_Pilot(sampleRate, pilotCodeLength,codeFreqBasis,  np.array([prnId]))
                codeTableSampCnt_pilot = len(codeTable_pilot)
                
                status, codePhase, doppler = l1.navic_pcps_acquisition1(
                                            waveform, 
                                            codeTable_pilot[np.arange(0, samples_in_10ms)%codeTableSampCnt_pilot, 0],
                                            codeTable_data[np.arange(0, samples_in_10ms)%codeTableSampCnt_data, 0],
                                            sampleRate ,fSearch,1,2, samples_in_10ms,if_frequency,12,False
                                        )   
                
                delaySamp = codePhase
                codePhase = (codePhase % codeTableSampCnt_data)/(sampleRate/codeFreqBasis)
            
                print(f"Acquisition results for PRN ID {prnId}\n Status:{status} Doppler:{doppler} Delay/Code-Phase:{delaySamp}/{codePhase}")

                # If a satellite is visible, initialize tracking loop
                if(status == True):
                    satVis += 1  
                    sat= np.append(sat,prnId)
                    tracker.append(l1.NavicTracker(prnId))
                    tracker[-1].SampleRate = sampleRate
                    tracker[-1].CenterFrequency = if_frequency
                    tracker[-1].PLLNoiseBandwidth = PLLNoiseBandwidth
                    tracker[-1].FLLNoiseBandwidth = FLLNoiseBandwidth
                    tracker[-1].DLLNoiseBandwidth = DLLNoiseBandwidth
                    tracker[-1].PLLIntegrationTime = round(PLLIntegrationTime*1e3)
                    tracker[-1].PRNID = prnId
                    tracker[-1].InitialDopplerShift = doppler
                    tracker[-1].InitialCodePhaseOffset = delaySamp#samples_in_10ms - delaySamp
                    tracker[-1].Correlators = no_of_correlators
                    tracker[-1].setupImpl(dataCodeLength, bufsize_power_estimation, cn0_min, detector_threshold, lock_fail_counter_threshold)
                    tracker[-1].samp_20msec_buff = np.copy(waveform)
                    tracker[-1].resetImpl()
                    tracker[-1].sample_count = delaySamp
                    nav_data.append(navdata.navic_l1_navigation_data(numSteps)) # initialize Nav Bits decode class
                    
            trackDataShape = (round(numSteps), satVis)
            y_pilot = np.empty(trackDataShape, dtype=np.complex_)
            y_data = np.empty(trackDataShape, dtype=np.complex_)
            fqyerr = np.empty(trackDataShape)
            fqynco = np.empty(trackDataShape)
            pherr = np.empty(trackDataShape)
            phnco = np.empty(trackDataShape)
            delayerr = np.empty(trackDataShape)
            delaynco = np.empty(trackDataShape)
            cn0_cap = np.empty(trackDataShape)
            pli = np.empty(trackDataShape)
            lock_fail_counter = np.empty(trackDataShape)
            prevPhase = np.empty(trackDataShape)
            fc = np.empty(trackDataShape)
            currnsamp = np.empty(trackDataShape)
            track_rem_code = np.empty(trackDataShape)

    incr_tow += 0.01 # update tow for each iteration
    # Perform tracking for visible satellites
    waveform = read_from_file(file,samples_in_10ms,bytes_per_sample)
    for i in range(satVis): 
        currnsamp[istep,i] , y_pilot[istep, i], y_data[istep,i],fqyerr[istep, i], fqynco[istep, i], pherr[istep, i], phnco[istep, i], delayerr[istep, i], delaynco[istep, i], cn0_cap[istep,i], pli[istep,i], lock_fail_counter [istep,i],fc[istep,i], prevPhase[istep,i] = tracker[i].stepImpl(waveform,istep)
        # print("istep=",istep, currnsamp[istep,i])
        tracker[i].sample_count += tracker[i].currnsamp
        remcode = tracker[i].old_remcode_pilot * tracker[i].SampleRate / tracker[i].codeFreq
        nav_data[i].obs_data.addval(incr_tow, tracker[i].sample_count, istep,remcode)
    
np.set_printoptions(precision=5,suppress = True)           


k = len(y_data)    
if len(sat)==0:
    print("No satellites were detected.")
else:
    sat=sat.astype(np.int16)
        
print("\n")
np.set_printoptions(threshold=np.inf)


for i in range(satVis):
    
    pilot_overlay_local_gen = l1.genNavicCaCode_Pilot_Overlay(sat[i])
    pilot_overlay_local_gen = pilot_overlay_local_gen.astype(int)
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
    #plt.plot(pilotOverlay_received)
    #plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Delay NCO')
    
    
    plt.savefig('./myplot.png')
    plt.show()
    
    
    n = 1 #Number of data per bit
    skip = 0 #Forgo few bits as the tracking loops starts early
    
    #Plotting tracking output 
    plt.scatter(np.real(y_data[:,i]),np.imag(y_data[:,i]),label='Pilot overlay signals detected',color='blue')
    plt.xlabel('Real') ; plt.ylabel('Imag')
    plt.scatter(np.real(y_pilot[:,i]),np.imag(y_pilot[:,i]),label='Data signals detected', color='orange')
    # Axis labels with increased font size
    plt.xlabel('Real', fontsize=18)
    plt.ylabel('Imag', fontsize=18)
    # Legend with larger font
    plt.legend(fontsize=18)
    # Grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Save and show
    plt.savefig('./bits.png')
    plt.show()


    databits_received = np.real(y_data[n*skip:,i])
    pilotOverlay_received = np.imag(y_pilot[n*skip:,i])
    
    mapbits = lambda l: np.piecewise(l, [l < 0, l >= 0], [1, 0])
    mapbits_inverted = lambda l: np.piecewise(l, [l < 0, l >= 0], [0, 1])
    bits = mapbits(databits_received)
    overlay = mapbits(pilotOverlay_received)
    overlay = overlay.astype(int)
    corr_output = signal.correlate(overlay,pilot_overlay_local_gen,mode='valid')
    max_corr_out = (np.argmax(corr_output))
    print(corr_output[max_corr_out], max_corr_out)

    bits_inverted = mapbits_inverted(databits_received)
    overlay_inverted = mapbits_inverted(pilotOverlay_received)
    overlay_inverted = overlay_inverted.astype(int)
    corr_inverted_output = signal.correlate(overlay_inverted,pilot_overlay_local_gen,mode='valid')
    max_corr_inverted_out = (np.argmax(corr_inverted_output))
    print(corr_inverted_output[max_corr_inverted_out], max_corr_inverted_out)

    if (np.abs(corr_output[max_corr_out])>=np.abs(corr_inverted_output[max_corr_inverted_out])):
        overlay = overlay[max_corr_out:1800+max_corr_out]
        bits = bits_inverted[max_corr_out:1800+max_corr_out]
        print("Bits not inverted")
        nav_data[i].obs_data.first_subframe_cnt = max_corr_out
    else:
        overlay = overlay_inverted[max_corr_inverted_out:1800+max_corr_inverted_out]
        bits = bits[max_corr_inverted_out:1800+max_corr_inverted_out]
        nav_data[i].obs_data.first_subframe_cnt = max_corr_inverted_out
        print("Bits inverted")

    print("#of true overlay bits=",np.sum(pilot_overlay_local_gen==overlay))
#    
    decode=l1.decoder()
    
    s1_decoded,s2_decoded,s3_decoded=decode.subframes_decode(bits)
    
    print("Subframe1 After decoding (9 bits):",s1_decoded)

    print("Subframe2 After decoding (600 bits):",s2_decoded)

    #test code for crc

    subframe2_decoded_string = ''.join(str(x) for x in s2_decoded)
    subframe2_decoded_int = int(subframe2_decoded_string,2)
    subframe2_decoded_hex= hex(subframe2_decoded_int)
    sub = ''.join(str(e) for e in subframe2_decoded_hex)
    subframe2decodedint = int(sub,16)
        
    decode.printlongdiv(subframe2decodedint, 25578747)
 

    print("Subframe3 After decoding (274 bits):",s3_decoded)

    subframe3_decoded_string = ''.join(str(x) for x in s3_decoded)
    subframe3_decoded_int = int(subframe3_decoded_string,2)
    subframe3_decoded_hex= hex(subframe3_decoded_int)
    sub3 = ''.join(str(e) for e in subframe3_decoded_hex)
    subframe3decodedint = int(sub3,16)
        
    decode.printlongdiv(subframe3decodedint, 25578747)
    
    
    nav_data[i].decode_nav_bits(s1_decoded,s2_decoded) # Decode the Navigation bits
    
    #ITOW indicates 2 hour period starting from the week while TOI indictes # of seconds elapsed within 2 hours.
    #nav_data[i].obs_data.first_subframe_tow = nav_data[i].toi + nav_data[i].itow*7200  # 7200 = 2 * 3600 (# of seconds in an hour)
    ##Here, we are updating the TOW for last 100 values in Observables buffer , to compute Pseudorange
    
    for j in range(numSteps-100,numSteps):  
       nav_data[i].obs_data.observables_fifo_buffer[j].obs_time_of_week = nav_data[i].toi + nav_data[i].itow*7200 -18 + (j -  nav_data[i].obs_data.first_subframe_cnt)*0.01
    


reftow_indeces = np.zeros(satVis, dtype = np.int32)
cumulative_sample_count_arr = np.zeros(satVis)
rem_code_arr  = np.zeros(satVis)


#compute reftow
reftow = 3600*24*7

#find the minimum latest tow among all satellites
for i in range(satVis):
    if (nav_data[i].obs_data.observables_fifo_buffer[numSteps-1].obs_time_of_week < reftow):
        reftow = nav_data[i].obs_data.observables_fifo_buffer[numSteps-1].obs_time_of_week
        
#find the indexes in the observation arrays corresponds to the nearest tow value with respect to the computed reftow above
for i in range(satVis):
    for j in range(80):
        if((nav_data[i].obs_data.observables_fifo_buffer[numSteps-1-80-j].obs_time_of_week - reftow)  < 1e-5):
            reftow_indeces[i] = numSteps-80+j
            break

reference_satelite = 0
min_sample_cnt = (1<<63)

#find the reference satellite (satellite that contains minimum sample count among all satellites)
for i in range(satVis):
    
    cumulative_sample_count_arr[i] = nav_data[i].obs_data.observables_fifo_buffer[reftow_indeces[i]].cumulative_sample_count
    rem_code_arr[i]                = nav_data[i].obs_data.observables_fifo_buffer[reftow_indeces[i]].remaining_codePhase
    myindex = nav_data[i].obs_data.first_subframe_cnt
    cum_count_at_frame_index = nav_data[i].obs_data.observables_fifo_buffer[myindex].cumulative_sample_count
    rem_code_at_frame_index = nav_data[i].obs_data.observables_fifo_buffer[myindex].remaining_codePhase
    
    print("cumcount = ",cumulative_sample_count_arr[i],"remcode at reftom=",rem_code_arr[i])
    print("cumcount at frameindex = ",cum_count_at_frame_index,"remcode at frameindex = ", rem_code_at_frame_index)
    if(nav_data[i].obs_data.observables_fifo_buffer[reftow_indeces[i]].cumulative_sample_count < min_sample_cnt):
        reference_satelite = i
        min_sample_cnt = nav_data[i].obs_data.observables_fifo_buffer[reftow_indeces[i]].cumulative_sample_count
        
print("ref satellite=",reference_satelite)       
val =  nav_data[reference_satelite].obs_data.first_subframe_cnt +1800  # Frame Index of the reference Satellite
#Find out Cumulative Sample count at Frame Index for reference satellite
cumulative_sample_count_at_frame_index = nav_data[reference_satelite].obs_data.observables_fifo_buffer[val].cumulative_sample_count 

cum_count_difference = nav_data[reference_satelite].obs_data.observables_fifo_buffer[reftow_indeces[reference_satelite]].cumulative_sample_count - cumulative_sample_count_at_frame_index
sample_count_reference = cumulative_sample_count_at_frame_index + cum_count_difference - samples_in_10ms *11.9875685


print("val=",val,"cum_sam_at_frame=", cumulative_sample_count_at_frame_index, "samp count ref=",sample_count_reference)

#find the psuedorange for each satellite
for i in range(satVis):
    nav_data[i].psuedorange = (SPEED_OF_LIGHT/sampleRate) * (cumulative_sample_count_arr[i] - sample_count_reference - rem_code_arr[i])
    print("---------------------------------------------------------------------------")
    print(f" for {sat[i]} sv psuedo range = {nav_data[i].psuedorange}")
    print("---------------------------------------------------------------------------")

print("reftow=", reftow)
        
