import numpy as np
import math

SPEED_OF_LIGHT  = 299792458.0
#sampling_rate   = 4096000
A_ref = 42164200
mu = 3.986005e14
Omega_dot_e = 7.2921151467e-5
ELEVATION_ANGLE_THRESHOLD = 0.26179938779914941


WGS84_A = 6378137.0  # Earth's semi-major axis (meters)
FE_WGS84 = 1 / 298.257223563  # Earth's flattening
E2 = FE_WGS84 * (2 - FE_WGS84) 



def binary_to_decimal(binary_list, signed=True):
    n = len(binary_list)  
    binary_str = "".join(map(str, binary_list))
    if signed and binary_list[0] == 1:  
        decimal_value = -((1 << (n - 1)) - int(binary_str[1:], 2))
    else:  
        decimal_value = int(binary_str, 2)

    return decimal_value


class navic_l1_navigation_data:
    def __init__(self,subframe_1,subframe_2):
        
        #subframe 1
        self.toi = binary_to_decimal(subframe_1, signed=False)*18
        
        #subframe 2
        self.week_no          =                binary_to_decimal(subframe_2[0:13], signed=False)
        self.itow             =                binary_to_decimal(subframe_2[13:21], signed=False)
        self.alert            =                subframe_2[21]
        self.l1_sps_health    =                subframe_2[22]
        self.iod              =                binary_to_decimal(subframe_2[23:27], signed=False)
        self.urai             =                binary_to_decimal(subframe_2[27:32], signed=True)
        self.t_oec            =                binary_to_decimal(subframe_2[32:43], signed=False)*300
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
        
        
        
        self.pseudorange         =                0
        self.receive_time_uncorr =                0
        self.tr_time_uncorr      =                0
        self.tr_time_corr        =                0
        self.delta_t_sv          =                0
        
        #satellite ecef coordinates
        self.ecef_x = 0
        self.ecef_y = 0
        self.ecef_z = 0
        
    
    def compute_ecc_anamoly(self,t):
        iterations = 200
        delta = 10e-10
        tk = t - self.t_oec
        if(tk > 302400):
            tk = tk - 604800
        if(tk < -302400):
            tk = tk + 604800
        
        semi_major_axis = A_ref + self.delta_A + self.a_dot*tk
        delta_n = self.delta_n_0 + (self.delta_n_dot*tk)/2
        n0 = np.sqrt(mu/semi_major_axis)/semi_major_axis
        n = n0 + delta_n
        M_k = self.M_0 + n*tk
        
        if self.e < 0.8:
            E_k = M_k
            
        else:
            math.pi
            
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
        self.delta_t_sv  =   self.af0 + self.af1* delta_t + self.af2*(delta_t**2) + delta_tr  
        self.tr_time_corr = self.tr_time_uncorr -  self.delta_t_sv


    def compute_satellite_positions(self):
        tk = self.tr_time_corr - self.t_oec
        if (tk > 302400):
            tk -= 604800
        if (tk < -302400):
            tk += 604800
            
        semi_major_axix = A_ref + self.delta_A + self.a_dot*tk
        E_k = self.compute_ecc_anamoly(self.tr_time_corr)
        
        true_anamoly = np.arctan2(((np.sqrt(1 - self.e**2) * np.sin(E_k))/(1-self.e*np.cos(E_k))),
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
        
        Omega_k  =   self.Omega_0 + (self.Omega_dot - Omega_dot_e)*tk - Omega_dot_e * self.t_oec
        
        self.ecef_x = xk_1 * np.cos(Omega_k) - yk_1*np.cos(i_k) * np.sin(Omega_k)
        self.ecef_y = xk_1 * np.sin(Omega_k) + yk_1*np.cos(i_k) * np.cos(Omega_k)
        self.ecef_z = yk_1 * np.sin(i_k)
        
        print("---------------------------------------------------------------------------------------------------------------------")
        print(f"  PRN ID = {self.prnid} x = {self.ecef_x} y = {self.ecef_y} z = {self.ecef_z} Pseudo range = {self.pseudorange}")
        
            
        
        




def ecef_to_latlong(r):
    r2 = r[0]**2 + r[1]**2  
    v = WGS84_A
    z=r[2]
    zk = 0.0

    while abs(z - zk) >= 1e-4:  
        zk = z
        sinp = z / np.sqrt(r2 + z**2)
        v = WGS84_A / np.sqrt(1 - E2 * sinp**2)
        z = r[2] + v * E2 * sinp

    pos = np.zeros(3)
    pos[0] = np.arctan(z / np.sqrt(r2)) if r2 > 1e-12 else (np.pi / 2 if r[2] > 0 else -np.pi / 2)
    pos[1] = np.arctan2(r[1], r[0]) if r2 > 1e-12 else 0.0
    pos[2] = np.sqrt(r2 + z**2) - v

    return pos


def ecef_to_enu(pos, r):
    sinp, cosp = np.sin(pos[0]), np.cos(pos[0])
    sinx, cosx = np.sin(pos[1]), np.cos(pos[1])

    e = np.array([
        -sinx * r[0] + cosx * r[1],
        -sinp * cosx * r[0] - sinp * sinx * r[1] + cosp * r[2],
        cosp * cosx * r[0] + cosp * sinx * r[1] + sinp * r[2]
    ])
    return e



def sat_az_el(pos, e):
    az, el = 0.0, np.pi / 2
    if pos[2] > -6378137.0:  # WGS84_A
        enu = ecef_to_enu(pos, e)
        enu_norm2 = enu[0]**2 + enu[1]**2

        az = np.arctan2(enu[0], enu[1]) if enu_norm2 >= 1e-12 else 0.0
        if az < 0:
            az += 2 * np.pi
        el = np.arcsin(enu[2])

    return np.array([az, el])


    
def compute_receiver_position(nav_data,no_of_channels):

    rx_ecef = np.zeros(3)
    unit_vector_bw_tx_rx = np.zeros(3)
    azel = np.zeros(3)
    time = 0
    
    
    
    max_iters = 100
    
    sv_ecef_x = np.zeros(no_of_channels)
    sv_ecef_y = np.zeros(no_of_channels)
    sv_ecef_z = np.zeros(no_of_channels)
    sv_time = np.zeros(no_of_channels)
    dpr = np.zeros(no_of_channels)
    H_matrix = np.zeros((no_of_channels, 4))    

    for i in range(no_of_channels):
        sv_ecef_x[i] = nav_data[i].ecef_x
        sv_ecef_y[i] = nav_data[i].ecef_y
        sv_ecef_z[i] = nav_data[i].ecef_z
        sv_time[i]   = nav_data[i].tr_time_corr
    
    for j in range(max_iters):
        # print(j)
        lat_long = ecef_to_latlong(rx_ecef)
        valid_sv = 0
        excluded_sv = -1*np.ones(no_of_channels)
        for ch in range(no_of_channels):
            
            geometric_range = np.sqrt((rx_ecef[0] - sv_ecef_x[ch])**2 +  
                                      (rx_ecef[1] - sv_ecef_y[ch])**2 + 
                                      (rx_ecef[2] - sv_ecef_z[ch])**2 )  
            
            geometric_range += Omega_dot_e * (rx_ecef[1] * sv_ecef_x[ch]  - sv_ecef_y[ch] * rx_ecef[0])/SPEED_OF_LIGHT
            
            
            unit_vector_bw_tx_rx[0] = (sv_ecef_x[ch] - rx_ecef[0])/geometric_range
            unit_vector_bw_tx_rx[1] = (sv_ecef_y[ch] - rx_ecef[1])/geometric_range     
            unit_vector_bw_tx_rx[2] = (sv_ecef_z[ch] - rx_ecef[2])/geometric_range
            
            azel = sat_az_el(lat_long, unit_vector_bw_tx_rx)
            #Filtering the satellites based on the elevation angle 
            if (azel[1] < ELEVATION_ANGLE_THRESHOLD):
            
                excluded_sv[ch] = ch
                #print(f" ***************** for sat = {nav_data[ch].prnid}  elevation_angle = {azel[1]} ****************\n")
                continue
            
            
            #print(f"j = {j}, ch = {ch} ,azel = {azel}")

            
            
            receive_time_uncorr = nav_data[ch].receive_time_uncorr
            
            #compute ionospheric and troposhperic corrections later
            
            #Form Hmatrix
            if ( ch != excluded_sv[ch]) :
                dpr[ch]     =  (nav_data[ch].pseudorange - SPEED_OF_LIGHT * nav_data[ch].Tgd) - (geometric_range + time - SPEED_OF_LIGHT * nav_data[ch].delta_t_sv) 
                H_matrix[ch][0] = -unit_vector_bw_tx_rx[0]
                H_matrix[ch][1] = -unit_vector_bw_tx_rx[1]
                H_matrix[ch][2] = -unit_vector_bw_tx_rx[2]
                H_matrix[ch][3] = 1
                valid_sv += 1
        
        if (valid_sv < 4):
            continue
        
        #compute H.T@H
        A = np.transpose(H_matrix) @ H_matrix
        b = np.transpose(H_matrix)@dpr
        x = np.linalg.solve(A,b)
        
        dx = x[0]
        dy = x[1]
        dz = x[2]
        dt = x[3]
        
        error = np.linalg.norm(x)
        
        if(error < 1e-8):
            break
        
        rx_ecef[0] += dx
        rx_ecef[1] += dy
        rx_ecef[2] += dz
        
        time += dt
    
    return rx_ecef
        


#sv 10 real time
subframe_1_10 = np.loadtxt("subframe1_10.txt").astype(int)
subframe_2_10 = np.loadtxt("subframe2_10.txt").astype(int)

# #sv 12
subframe_1_12 = np.loadtxt("subframe1_12.txt").astype(int)
subframe_2_12 = np.loadtxt("subframe2_12.txt").astype(int)

# #sv 13
subframe_1_13 = np.loadtxt("subframe1_13.txt").astype(int)
subframe_2_13 = np.loadtxt("subframe2_13.txt").astype(int)

# #sv 14
subframe_1_14 = np.loadtxt("subframe1_14.txt").astype(int)
subframe_2_14 = np.loadtxt("subframe2_14.txt").astype(int)


#sv 15
subframe_1_15 = np.loadtxt("subframe1_15.txt").astype(int)
subframe_2_15 = np.loadtxt("subframe2_15.txt").astype(int)


#sv 16
subframe_1_16 = np.loadtxt("subframe1_16.txt").astype(int)
subframe_2_16 = np.loadtxt("subframe2_16.txt").astype(int)

#sv 17
subframe_1_17 = np.loadtxt("subframe1_17.txt").astype(int)
subframe_2_17 = np.loadtxt("subframe2_17.txt").astype(int)
        
nav_data = []
#compute satellite positions
'''

---------------------------------------------------------------------------
 for 10 sv psuedo range = 37852765.858575955
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 12 sv psuedo range = 39782522.71080665
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 13 sv psuedo range = 37888360.793399975
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 14 sv psuedo range = 35937853.511898875
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 15 sv psuedo range = 38387999.77886196
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 16 sv psuedo range = 38302996.88171526
---------------------------------------------------------------------------
---------------------------------------------------------------------------
 for 17 sv psuedo range = 38052251.90992328
---------------------------------------------------------------------------
reftow= 26657.99

'''

pseudo_ranges = np.array([37852765.858575955,39782522.71080665,37888360.793399975,35937853.511898875,
                          38387999.77886196,38302996.88171526,38052251.90992328])



nav_data.append(navic_l1_navigation_data(subframe_1_10,subframe_2_10))
nav_data.append(navic_l1_navigation_data(subframe_1_12,subframe_2_12))
nav_data.append(navic_l1_navigation_data(subframe_1_13,subframe_2_13))
nav_data.append(navic_l1_navigation_data(subframe_1_14,subframe_2_14))
nav_data.append(navic_l1_navigation_data(subframe_1_15,subframe_2_15))
nav_data.append(navic_l1_navigation_data(subframe_1_16,subframe_2_16))
nav_data.append(navic_l1_navigation_data(subframe_1_17,subframe_2_17))

reftow = 26657.99

print("\n********************************** Satellite ECEF Positions and Pueudo ranges **********************************\n")
for i in range(7):
    #if (i != 1 or i!= 4):
    #if (i != 4):
        nav_data[i].pseudorange = pseudo_ranges[i]
        #nav_data[i].receive_time_uncorr = reftow + 120.083/1000
        #nav_data[i].receive_time_uncorr = reftow + 119.8962/1000
        nav_data[i].receive_time_uncorr = reftow + 119.875685/1000
        nav_data[i].tr_time_uncorr =  nav_data[i].receive_time_uncorr - (nav_data[i].pseudorange/SPEED_OF_LIGHT)
        nav_data[i].apply_time_corrections()
        nav_data[i].compute_satellite_positions()
    

rx_ecef = compute_receiver_position(nav_data,7)   

lat_long_alt =  ecef_to_latlong(rx_ecef)
lat = np.degrees(lat_long_alt[0])
lon = np.degrees(lat_long_alt[1])
height = lat_long_alt[2]

#print(f"tow = {reftow},lat = {lat} , lon = {lon} height = {height}")
print("\n******************** User Position **********************")
print (f"  Lat = {lat} N , Lon = {lon} E " )
print("*********************************************************")
#The Receiver location considered is  Lat: 23 degree North, Long: 72 degree East Alitude: 0 m
#lat = 23.603016454684 , lon = 73.56584090490936 height = -307674.490280075

        
