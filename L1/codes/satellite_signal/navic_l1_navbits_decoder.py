import numpy as np
import math
from collections import deque
from dataclasses import dataclass



SPEED_OF_LIGHT  = 299792458.0
#sampling_rate   = 4096000
A_ref = 42164200
mu = 3.986005e14
Omega_dot_e = 7.2921151467e-5



def binary_to_decimal(binary_list, signed=True):
    n = len(binary_list)  
    binary_str = "".join(map(str, binary_list)) 
    if signed and binary_list[0] == 1:  
        decimal_value = -((1 << (n - 1)) - int(binary_str[1:], 2))
    else:  
        decimal_value = int(binary_str, 2)

    return decimal_value


# Define the structure for Obsservables
@dataclass
class DataStruct:
    obs_time_of_week: np.float64  # double
    cumulative_sample_count: np.int64 # int
    iteration_counter: np.int64  # int
    remaining_codePhase : np.float64
    
    
class observables_data :
    def __init__(self,observation_count):
        # Create a FIFO buffer with a max size of 5
        self.observables_fifo_buffer  = deque(maxlen=observation_count)
        #self.first_subframe_tow = 0
        self.first_subframe_cnt = 0   
    def addval (self, tow, sample_cnt, loop_count, codephase):
        self.observables_fifo_buffer.append(DataStruct(tow,sample_cnt,loop_count,codephase))



class navic_l1_navigation_data:
    def __init__(self,count):
        self.obs_data = observables_data(count)  # Initialize Observables Data
    def decode_nav_bits(self,subframe_1,subframe_2):
        
        #subframe 1
        self.toi = binary_to_decimal(subframe_1, signed=False)*18
        
        #subframe 1
        self.week_no          =                binary_to_decimal(subframe_2[0:13], signed=False)
        self.itow             =                binary_to_decimal(subframe_2[13:21], signed=False)
        self.alert            =                subframe_2[21]
        self.l1_sps_health    =                subframe_2[22]
        self.iod              =                binary_to_decimal(subframe_2[23:27], signed=False)
        self.urai             =                binary_to_decimal(subframe_2[27:32], signed=False)
        self.t_oec            =                binary_to_decimal(subframe_2[32:43], signed=True)*300
        self.delta_A          =                binary_to_decimal(subframe_2[43:69], signed=True)*(2**(-9))
        self.a_dot            =                binary_to_decimal(subframe_2[69:95], signed=True)*(2**(-21))
        self.delta_n_0        =                binary_to_decimal(subframe_2[95:114], signed=True)*(2**(-44))*np.pi
        self.delta_n_dot      =                binary_to_decimal(subframe_2[114:137], signed=True)*(2**(-57))*np.pi
        self.M_0              =                binary_to_decimal(subframe_2[137:170], signed=True)*(2**(-32))*np.pi
        self.e                =                binary_to_decimal(subframe_2[170:203], signed=False)*(2**(-34))
        self.w                =                binary_to_decimal(subframe_2[203:236], signed=True)*(2**(-32))*np.pi
        self.Omega_0          =                binary_to_decimal(subframe_2[236:269], signed=True)*(2**(-32))*np.pi
        self.Omega_dot        =                binary_to_decimal(subframe_2[269:294], signed=True)*(2**(-44))*np.pi
        self.i_0              =                binary_to_decimal(subframe_2[294:327], signed=True)*(2**(-32))*np.pi
        self.idot             =                binary_to_decimal(subframe_2[327:342], signed=True)*(2**(-44))*np.pi
        self.c_is             =                binary_to_decimal(subframe_2[342:358], signed=True)*(2**(-30))
        self.c_ic             =                binary_to_decimal(subframe_2[358:374], signed=True)*(2**(-30))
        self.c_rs             =                binary_to_decimal(subframe_2[374:398], signed=True)*(2**(-8))
        self.c_rc             =                binary_to_decimal(subframe_2[398:422], signed=True)*(2**(-8))
        self.c_us             =                binary_to_decimal(subframe_2[422:443], signed=True)*(2**(-30))
        self.c_uc             =                binary_to_decimal(subframe_2[443:464], signed=True)*(2**(-30))
        self.af0              =                binary_to_decimal(subframe_2[464:493], signed=True)*(2**(-35))
        self.af1              =                binary_to_decimal(subframe_2[493:515], signed=True)*(2**(-50))
        self.af2              =                binary_to_decimal(subframe_2[515:530], signed=True)*(2**(-66))
        self.Tgd              =                binary_to_decimal(subframe_2[530:542], signed=True)*(2**(-35))
        self.isc_l1p          =                binary_to_decimal(subframe_2[542:554], signed=True)*(2**(-35))
        self.isc_l1d          =                binary_to_decimal(subframe_2[554:566], signed=True)*(2**(-35))
        self.rsf              =                subframe_2[566]
        self.spare            =                binary_to_decimal(subframe_2[567:570], signed=False)
        self.prnid            =                binary_to_decimal(subframe_2[570:576], signed=False)
        
        
        self.psuedorange         =                0
        self.receive_time_uncorr =                0
        self.tr_time_uncorr      =                0
        self.tr_time_corr        =                0
    
    
    def compute_psuedorange_and_time_uncorrected(self, sampling_rate):
        
        
        #compute reftow
        reftow = 3600*24*7
        
        
        
        
        
        #may be for navic 68.802 is not valid
        self.receive_time_uncorr = reftow + 68.802 / 1000.0
        self.tr_time_uncorr      = self.receive_time_uncorr - (self.psuedorange/SPEED_OF_LIGHT)
    
    def compute_ecc_anamoly(self,t):
        iterations = 200
        delta = 10e-10
        tk = t - self.t_oec
        if(tk > 302400):
            tk = tk - 604800
        if(tk < -302400):
            tk = tk + 604800
        
        semi_major_axix = A_ref + self.delta_A + self.a_dot*tk
        delta_n = self.delta_n_0 + (self.delta_n_dot*tk)/2
        n0 = np.sqrt(mu/semi_major_axix)/semi_major_axix
        n = n0 + delta_n
        M_k = self.M_0 + n*tk
        
        E_k = M_k if self.e < 0.8 else math.pi
        correction = E_k - (M_k + self.e * math.sin(M_k))

        # Solve iteratively
        while np.abs(correction) > delta and iterations > 0:
            last = E_k
            E_k = M_k + self.e * np.sin(E_k)
            correction = E_k - last
            iterations -= 1

        return E_k
        
             
    def apply_time_corrections(self):
        delta_t = self.tr_time_uncorr - self.t_oec
        if (delta_t > 302400):
            delta_t -= 604800
        if (delta_t < -302400):
            delta_t += 604800
    
        E_k = self.compute_ecc_anamoly(self.tr_time_uncorr)
        A0 = A_ref + self.delta_A
        delta_tr = -4.442807633e-10 * self.e * np.sqrt(A0)*np.sin(E_k)
        delta_t_sv  =   self.af0 + self.af1* delta_t + self.af2*(delta_t**2) + delta_tr  
        self.tr_time_corr = self.tr_time_uncorr - delta_t_sv


    def compute_satellite_positions(self):
        tk = self.tr_time_corr - self.t_oec
        if (tk > 302400):
            tk -= 604800
        if (tk < -302400):
            tk += 604800
            
        semi_major_axix = A_ref + self.delta_A + self.a_dot*tk
        E_k = self.compute_ecc_anamoly(self.tr_time_corr)
        
        true_anamoly = np.atan2(((np.sqrt(1 - self.e**2) * np.sin(E_k))/(1-self.e*np.cos(E_k))),
                                (np.cos(E_k) - self.e)/(1-self.e*np.cos(E_k)))
        
        phi_k = true_anamoly + self.w
        
        del_uk = self.c_us * np.sin(2*phi_k)    +    self.c_uc * np.cos(2*phi_k)
        del_rk = self.c_rs * np.sin(2*phi_k)    +    self.c_rc * np.cos(2*phi_k)
        del_ik = self.c_is * np.sin(2*phi_k)    +    self.c_ic * np.cos(2*phi_k)
        
        u_k = phi_k + del_uk
        r_k = semi_major_axix*(1-self.e*np.cos(E_k)) + del_rk
        i_k = self.i_0 + del_ik + self.idot*tk
        
        xk_1 = r_k * np.cos(u_k)
        yk_1 = r_k * np.sin(u_k)
        
       # Omega_k  =   self.Omega_0 +  
        





        
        
        
        