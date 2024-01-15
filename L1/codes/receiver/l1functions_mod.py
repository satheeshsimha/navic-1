import numpy as np
from fractions import Fraction
import math
import scipy.constants as sciconst
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import scipy.signal as signal
from sk_dsp_comm import fec_conv as fec     #pip/pip3 install scikit-dsp-comm
from numpy_ringbuffer import RingBuffer
from random import *
from ldpc import bp_decoder
import struct
import csv
# PRN Sequence generation API
#R0 regiser initial parameter for Data signal


def csv2bin( csvfile = None, binfile = None ):
    with open(csvfile, 'r') as c:
        with open(binfile, 'wb') as b:
            csvreader = csv.reader(c, delimiter=',')
            count = 0
            for row in csvreader:
                for col in row:
                    # Little endian 16-bit integers (shorts)
                    b.write( struct.pack("<h", int(float(col.strip()))) )
                count += 1

def chunked_read( fobj, chunk_bytes = 4*1024 ):
    while True:
        data = fobj.read(chunk_bytes)
        if( not data ):
            break
        else:
            yield data
def bin2csv( binfile= None, csvfile= None, chunk_bytes = 4*1024 ):
    rx_iq_array=[]
    with open(binfile, 'rb') as b:
        count=0
        for data in chunked_read(b, chunk_bytes = chunk_bytes):
            count += len(data)
            for i in range(0, len(data), 4):
                sig_i, = struct.unpack('<h', data[i:i+2])
                sig_q, = struct.unpack('<h', data[i+2:i+4])
                sig_i=sig_i/2048.0
                sig_q=sig_q/2048.0			
                rx_iq_array.append(1*sig_i+1j*sig_q)
        rx_iq_array=np.array(rx_iq_array)
        return rx_iq_array

SV_L1_Data_r0 = {
    1: '0061727026503255544',
    2: '1660130752435362260',
    3: '0676457016477551225',
    4: '1763467705267605701',
    5: '1614265052776007236',
    6: '1446113457553463523',
    7: '1467417471470124574',
    8: '0022513456555401603',
    9: '0004420115402210365',
   10: '0072276243316574510',
   11: '1632356715721616750',
   12: '1670164755420300763',
   13: '1752127524253360255',
   14: '0262220014044243135',
   15: '1476157654546440020',
   16: '1567545246612304745',
   17: '0341667641424721673',
   18: '0627234635353763045',
   19: '0422600144741165152',
   20: '1661124176724621030',
   21: '1225124173720602330',
   22: '1271773065617322065',
   23: '0611751161355750124',
   24: '0121046615341766266',
   25: '0337423707274604122',
   26: '0246610305446052270',
   27: '0427326063324033344',
   28: '1127467544162733403',
   29: '0772425336125565156',
   30: '1652465113031101044',
   31: '1737622607214524550',
   32: '1621315362240732407',
   33: '0171733204500613155',
   34: '1462031354327077565',
   35: '1141265411761074755',
   36: '0665106277260231251',
   37: '0573123144343776027',
   38: '0222101406610314705',
   39: '0140673225434336401',
   40: '0624233245727625631',
   41: '0224022145647544263',
   42: '0222501602610354705',
   43: '1370337660412244327',
   44: '0563567347256715524',
   45: '1407636661116077143',
   46: '1137431557133151004',
   47: '1113003456475500265',
   48: '1746553632646152413',
   49: '1465416631251321074',
   50: '0130516430377202712',
   51: '0762173527246302776',
   52: '1606732407336425136',
   53: '1131112010066741562',
   54: '1107467740060732403',
   55: '0755500241327076744',
   56: '1443037764170374631',
   57: '0243224434357700345',
   58: '0445504023027564357',
   59: '1211152271373271472',
   60: '0256644102553071753',
   61: '0733312314424771412',
   62: '1636376400221406415',
   63: '0574114621235461516',
   64: '1710717574016037362',
   
}
#R1 regiser initial parameter for Data signal
SV_L1_Data_r1 = {
    1: '0377627103341647600',
    2: '0047555332635133703',
    3: '0570574070736102152',
    4: '0511013576745450615',
    5: '1216243446624447775',
    6: '0176452272675511054',
    7: '0151055342317137706',
    8: '1127720116046071664',
    9: '0514407436155575524',
   10: '0253070462740453542',
   11: '0573371306324706336',
   12: '1315135317732077306',
   13: '1170303027726635012',
   14: '1637171270537414673',
   15: '0342370520251732111',
   16: '0142423551056551362',
   17: '0641261355426453710',
   18: '0237176034757345266',
   19: '1205663360515365064',
   20: '0725000004121104102',
   21: '0337367500320303262',
   22: '1303374445022536530',
   23: '1033071464007363115',
   24: '0753124124237073577',
   25: '0133522075443754772',
   26: '1244212514312345145',
   27: '1066056211234322164',
   28: '0073115240113351010',
   29: '1102260031574577224',
   30: '1166703527236520553',
   31: '0056062273631723177',
   32: '0141517013160576212',
   33: '1644007677312431616',
   34: '0201757033615262622',
   35: '0357610362675720200',
   36: '1637504174727237065',
   37: '1510345507743707753',
   38: '0540160763721100120',
   39: '0406415410457500342',
   40: '0707515543554212732',
   41: '0140216674314371011',
   42: '0445414471314273300',
   43: '0120121661750263177',
   44: '0477301251340044262',
   45: '1157040657040363676',
   46: '1222265021477405004',
   47: '0314661556545362364',
   48: '0177320240371640542',
   49: '0735517310345570340',
   50: '1367565551220511432',
   51: '1274167141162675644',
   52: '1543641015130470077',
   53: '0640733734534576460',
   54: '0216312531021205434',
   55: '0050232164401566177',
   56: '0702636370401726111',
   57: '1733537351460015703',
   58: '1523265651140460620',
   59: '0607703231502460135',
   60: '1757246242710445777',
   61: '0464412467237572274',
   62: '1050617751566552643',
   63: '1041606123021052264',
   64: '1335441345250455042',
   
}
#C regiser initial parameter for Data signal
SV_L1_Data_C = {
    1: '10100',
    2: '10100',
    3: '00110',
    4: '10100',
    5: '10100',
    6: '00110',
    7: '10100',
    8: '00110',
    9: '00110',
   10: '00110',
   11: '10100',
   12: '00110',
   13: '10100',
   14: '00110',
   15: '00110',
   16: '10100',
   17: '00110',
   18: '00110',
   19: '00110',
   20: '00110',
   21: '10100',
   22: '10100',
   23: '10100',
   24: '00110',
   25: '10100',
   26: '00110',
   27: '00110',
   28: '00110',
   29: '00110',
   30: '10100',
   31: '10100',
   32: '00110',
   33: '10100',
   34: '00110',
   35: '00110',
   36: '00110',
   37: '10100',
   38: '10100',
   39: '01100',
   40: '00110',
   41: '00011',
   42: '01100',
   43: '10100',
   44: '00110',
   45: '10100',
   46: '10100',
   47: '00110',
   48: '00110',
   49: '00110',
   50: '10100',
   51: '10100',
   52: '10100',
   53: '00110',
   54: '10100',
   55: '00110',
   56: '10100',
   57: '00110',
   58: '00110',
   59: '10100',
   60: '10010',
   61: '10001',
   62: '11000',
   63: '00110',
   64: '10100',
   
}
#R0 regiser initial parameter for Pilot signal
SV_L1_Pilot_r0 = {
    1: '0227743641272102303',
    2: '0603070242564637717',
    3: '0746325144437416120',
    4: '0023763714573206044',
    5: '0155575663373106723',
    6: '0022277536552741033',
    7: '0137757627072411730',
    8: '0413034001670700216',
    9: '0501123675324707024',
   10: '0013727517464264567',
   11: '0663351450332761127',
   12: '1450710073416110356',
   13: '1716542347100366110',
   14: '0743601273016301212',
   15: '1454332372150500137',
   16: '1473215015316613621',
   17: '1255535602164437613',
   18: '1164537254033266174',
   19: '1500537251137244274',
   20: '0766727150471256024',
   21: '0457637114652202460',
   22: '0436500136253056124',
   23: '1666265767713037215',
   24: '1465272157164065443',
   25: '0607440357166466472',
   26: '1670202421463640077',
   27: '1312661744614412524',
   28: '1413034001672741216',
   29: '1113765722434040551',
   30: '0621573414133237134',
   31: '0526104310250410535',
   32: '0426454733176070600',
   33: '1440644676733136472',
   34: '0557275325702027456',
   35: '0657637150553356442',
   36: '1403560400557766512',
   37: '1531165662277124403',
   38: '1403072012721162611',
   39: '0541210077534050730',
   40: '1660256422576622574',
   41: '0646767375467672136',
   42: '1563301635027210017',
   43: '1403462012723163611',
   44: '0767233376550711053',
   45: '1260555130762307205',
   46: '0531075060147161624',
   47: '0112673710551347402',
   48: '1314750013607403146',
   49: '0471706447643213002',
   50: '0770352206645261362',
   51: '0255127616022236737',
   52: '1035616240477274125',
   53: '0251115713566666576',
   54: '0752241454312660541',
   55: '0461250256520434602',
   56: '1116341217327713444',
   57: '0765232132271554573',
   58: '0774370107303671123',
   59: '1407140711055577677',
   60: '1753355476331367516',
   61: '0101630163132222775',
   62: '0730471404057577456',
   63: '1336743247162047542',
   64: '0020666576373544533',
   
}
#R1 regiser initial parameter for Pilot signal
SV_L1_Pilot_r1 = {
    1: '1667217344450257245',
    2: '0300642746017221737',
    3: '0474006332201753645',
    4: '0613606702460402137',
    5: '1465531713404064713',
    6: '1063646422557130427',
    7: '1066060465055002004',
    8: '0225574416605070652',
    9: '1733560674073230405',
   10: '1116277147142260461',
   11: '0663351450332761127',
   12: '1110300535412261305',
   13: '1046105227571557243',
   14: '1020346561064461527',
   15: '1270052747201123510',
   16: '1041553307136735706',
   17: '1002352163603013730',
   18: '1362622514254366256',
   19: '0556645716623157361',
   20: '0020341533300021636',
   21: '1470231623730254774',
   22: '1437100574634755567',
   23: '0215346037247347710',
   24: '1074246275146357122',
   25: '1655552356143710472',
   26: '1067241424131022656',
   27: '1611144345044137740',
   28: '1235122601654653275',
   29: '0663754302501454556',
   30: '0330540311241344370',
   31: '1763277034331577303',
   32: '1325110610226320770',
   33: '0632344657312671631',
   34: '1432530060077160315',
   35: '1272177170234542346',
   36: '0043174152003062273',
   37: '0633575650312403065',
   38: '0305021033755066410',
   39: '0137373436464572225',
   40: '0014331642301151614',
   41: '0444423305436737401',
   42: '0232343171540161113',
   43: '0101411166154322757',
   44: '0501120665453153342',
   45: '1042475051720150775',
   46: '1533531265037673325',
   47: '0506620200211067675',
   48: '1324133406103765602',
   49: '0203136107415235456',
   50: '1521524233172031026',
   51: '0164213410044443204',
   52: '1221110757557452411',
   53: '0252317630101475044',
   54: '0014540074363706135',
   55: '0371711523526255275',
   56: '0012400567546521471',
   57: '0312622351062337705',
   58: '0023647344743400250',
   59: '0257310611765747211',
   60: '1540176212407214706',
   61: '1412637164262406706',
   62: '0363125736302421243',
   63: '0414175374460515677',
   64: '0004500310276201661',
   
}
#C regiser initial parameter for Pilot signal
SV_L1_Pilot_C = {
    1: '01000',
    2: '00000',
    3: '01000',
    4: '00000',
    5: '01000',
    6: '01000',
    7: '00000',
    8: '01000',
    9: '00000',
   10: '00000',
   11: '00000',
   12: '01000',
   13: '01000',
   14: '00000',
   15: '00000',
   16: '00000',
   17: '01000',
   18: '01000',
   19: '01000',
   20: '01000',
   21: '00000',
   22: '01000',
   23: '01000',
   24: '00000',
   25: '01000',
   26: '00000',
   27: '01000',
   28: '00000',
   29: '00000',
   30: '00000',
   31: '00000',
   32: '01000',
   33: '01000',
   34: '00000',
   35: '01000',
   36: '01000',
   37: '00110',
   38: '00000',
   39: '01010',
   40: '00110',
   41: '00101',
   42: '10001',
   43: '00110',
   44: '00000',
   45: '10001',
   46: '00000',
   47: '00110',
   48: '00101',
   49: '00110',
   50: '10010',
   51: '10001',
   52: '00011',
   53: '01000',
   54: '00000',
   55: '00000',
   56: '00101',
   57: '10001',
   58: '00000',
   59: '01000',
   60: '00000',
   61: '00000',
   62: '10001',
   63: '00000',
   64: '01000',
}
#R0 regiser initial parameter for Pilot Overlay signal
SV_L1_Pilot_Overlay_r0 = {
    1: '0110111011',
    2: '0111101000',
    3: '1100000001',
    4: '0110110110',
    5: '0100011000',
    6: '0011111100',
    7: '0011111100',
    8: '1111000101',
    9: '0011001100',
   10: '1000011010',
   11: '0001001001',
   12: '0110101011',
   13: '0101110000',
   14: '0010110011',
   15: '1110000111',
   16: '1000000000',
   17: '1111101101',
   18: '1111101011',
   19: '0010001011',
   20: '0011101000',
   21: '0011011010',
   22: '0011111100',
   23: '0111001100',
   24: '1000101110',
   25: '0101000010',
   26: '0000101010',
   27: '0000100001',
   28: '1000010000',
   29: '1011100100',
   30: '0110111111',
   31: '1001110000',
   32: '1101110101',
   33: '0101111100',
   34: '1011001000',
   35: '1000001100',
   36: '0001100101',
   37: '0000000010',
   38: '0010100011',
   39: '1111010010',
   40: '0000100101',
   41: '0100111011',
   42: '0110111001',
   43: '0010011101',
   44: '1000011010',
   45: '0010000010',
   46: '1001001111',
   47: '1111001111',
   48: '0010110010',
   49: '0111111110',
   50: '0100100011',
   51: '0100001110',
   52: '0111101101',
   53: '1000010010',
   54: '1001001110',
   55: '0001011110',
   56: '1110001001',
   57: '1110110001',
   58: '1101111110',
   59: '0111111000',
   60: '1010001111',
   61: '1100110100',
   62: '0011010010',
   63: '1101010100',
   64: '1001110110',
   
}
#R1 regiser initial parameter for Pilot Overlay signal
SV_L1_Pilot_Overlay_r1 = {
    1: '0100110000',
    2: '0110000010',
    3: '1110010001',
    4: '0101110011',
    5: '1011000110',
    6: '1010101111',
    7: '1110001000',
    8: '0001010000',
    9: '1011111100',
   10: '0100010101',
   11: '1100000100',
   12: '0111011110',
   13: '1001110011',
   14: '1001101010',
   15: '0001100101',
   16: '0101101000',
   17: '0111111011',
   18: '1001110001',
   19: '1101011001',
   20: '0111011110',
   21: '0011100101',
   22: '1101000001',
   23: '0110110001',
   24: '0011000001',
   25: '1111100001',
   26: '0010011011',
   27: '0110011110',
   28: '0000111000',
   29: '0000000101',
   30: '0000100100',
   31: '0110101101',
   32: '1011010001',
   33: '0001110111',
   34: '0110100111',
   35: '0111010101',
   36: '1110110101',
   37: '1011110110',
   38: '1011011010',
   39: '1100101010',
   40: '1101101111',
   41: '1110011111',
   42: '1000100000',
   43: '0110000101',
   44: '0101111101',
   45: '0011110111',
   46: '1010001010',
   47: '1101000011',
   48: '1101101101',
   49: '1011101001',
   50: '0100001100',
   51: '1001100010',
   52: '1100110011',
   53: '0011110101',
   54: '0100110100',
   55: '1110011000',
   56: '1000111100',
   57: '0100010000',
   58: '0010011101',
   59: '1100011010',
   60: '0010011000',
   61: '0001001000',
   62: '0110001110',
   63: '0110101101',
   64: '1100011011',
   
}
#initial condition of register G2 taken from NavIC ICD
SV_L5 = {
   1: '1110100111',
   2: '0000100110',
   3: '1000110100',
   4: '0101110010',
   5: '1110110000',
   6: '0001101011',
   7: '0000010100',
   8: '0100110000',
   9: '0010011000',
  10: '1101100100',
  11: '0001001100',
  12: '1101111100',
  13: '1011010010',
  14: '0111101010',
}

SV_S = {
   1: '0011101111',
   2: '0101111101',
   3: '1000110001',
   4: '0010101011',
   5: '1010010001',
   6: '0100101100',
   7: '0010001110',
   8: '0100100110',
   9: '1100001110',
  10: '1010111110',
  11: '1110010001',
  12: '1101101001',
  13: '0101000101',
  14: '0100001101',
}


# define a function to convert an octal digit to binary
def octal_to_binary(octal_digit):
    # define a dictionary to map octal digits to binary
    octal_to_binary_dict = {
        '0': '000',
        '1': '001',
        '2': '010',
        '3': '011',
        '4': '100',
        '5': '101',
        '6': '110',
        '7': '111'
    }
    # return the binary equivalent of the octal digit
    return octal_to_binary_dict[octal_digit]


# define a function to convert an octal bit string to binary
def octal_bits_to_binary(octal_bits):
    # remove the leftmost bit of the octal string
    leftmost_bit = octal_bits[0]
    remaining_bits = octal_bits[1:]
    # convert the remaining bits to binary and concatenate them
    binary_bits = ''
    for octal_digit in remaining_bits:
        binary_digit = octal_to_binary(octal_digit)
        binary_bits += binary_digit
    # add the leftmost bit back to the leftmost position of the binary result
    binary_result = leftmost_bit + binary_bits
    return binary_result

# iterate over the dictionary and convert each octal bit string to binary
for key, value in SV_L1_Data_r0.items():
    binary_value = octal_bits_to_binary(value)
    SV_L1_Data_r0[key] = binary_value
    #print(SV_L1_Data_r0[key])
for key, value in SV_L1_Data_r1.items():
    binary_value = octal_bits_to_binary(value)
    SV_L1_Data_r1[key] = binary_value
for key, value in SV_L1_Pilot_r0.items():
    binary_value = octal_bits_to_binary(value)
    SV_L1_Pilot_r0[key] = binary_value
for key, value in SV_L1_Pilot_r1.items():
    binary_value = octal_bits_to_binary(value)
    SV_L1_Pilot_r1[key] = binary_value    
    
    
#PRN code generation for NavIC constellation
#function to shift the bits according to taps given to registers


def shift(register, feedback, length):

    for i in range(length-1):
        register[i] = register[i+1]
    register[length-1] = feedback


def genNavicCaCode_Data(sv):
    """Build the PRN Sequence for Data signal for a given satellite ID
    
    :param int sv: satellite code (1-14 L5 band, 15-28 S band)
    :returns list: Data PRN sequence for chosen satellite
    
    """
    # init registers
    #G1 = [1 for i in range(10)]
    if(sv<1 or sv>64):
        print("Error: PRN ID out of bounds!")
        return None
    else:
        r0 = [int(i) for i in [*SV_L1_Data_r0[sv]]]
        r1 = [int(i) for i in [*SV_L1_Data_r1[sv]]]
        C  = [int(i) for i in [*SV_L1_Data_C[sv]]]
       
    
    
    cad = [] # stuff data output in here
    
    # create primary data sequence
    codeLength = 10230
    
    i=0
    for j in range(codeLength):
        #print(j)
        a = ((r0[50]^r0[45])^(r0[40]))^((r0[20]^r0[10])^(r0[5]^r0[0]))
        # compute σ2A
        sigma2A = ((r0[50]^r0[45])^r0[40]) & ((r0[20]^r0[10])^(r0[5]^r0[0]))
        # compute σ2B
        sigma2B = ((r0[50]^r0[45])&(r0[40]))^((r0[20]^r0[10])&(r0[5]^r0[0]))
        # compute σ2C
        sigma2C = (r0[50]&r0[45])^((r0[20]&r0[10])^(r0[5]&r0[0]))
        # compute σ2
        sigma2 = (sigma2A ^ sigma2B) ^ sigma2C
        # compute r1A
        temp= ((r0[40]^r0[35])^(r0[30]^r0[25]))^(r0[15]^r0[0])
        R1A = sigma2 ^ temp
        # compute r1B
        R1B = ((r1[50]^r1[45])^(r1[40]^r1[20]))^((r1[10]^r1[5])^(r1[0]))
        b  = R1A ^ R1B
        z  = r1[0]^C[0]
        shift(C,C[0],5) 
        shift(r1,b,55)
        shift(r0,a,55)
        
        cad.append((z))
    
    return  np.array(cad)   
 
def genNavicCaCode_Pilot(sv):
    """Build the PRN sequence for Pilot for a given satellite ID
    
    :param int sv: satellite code (1-64)
    :returns list: PRN Sequence for Pilot for chosen satellite
    
    """
     # init registers
    #G1 = [1 for i in range(10)]
    if(sv<1 or sv>64):
        print("Error: PRN ID out of bounds!")
        return None
    else:
        
        r0_p = [int(i) for i in [*SV_L1_Pilot_r0[sv]]]
        r1_p = [int(i) for i in [*SV_L1_Pilot_r1[sv]]]
        C_p  = [int(i) for i in [*SV_L1_Pilot_C[sv]]]
       
    
    
    cap = []# stuff piolt output in here
    
    # create primary data sequence
    codeLength = 10230
     
    for k in range(codeLength):
        r_p = (r0_p[50]^r0_p[45])^(r0_p[40])^((r0_p[20]^r0_p[10])^(r0_p[5]^r0_p[0]))
        #r_p = shift(r0_p, [51,46,41,21,11,6,1],[55])
        # compute σ2A
        sigma2A=(r0_p[50]^r0_p[45]^r0_p[40])&(r0_p[20]^r0_p[10]^r0_p[5]^r0_p[0])
        # compute σ2B
        sigma2B = ((r0_p[50]^r0_p[45])&(r0_p[40]))^((r0_p[20]^r0_p[10])&(r0_p[5]^r0_p[0]))
        # compute σ2C
        sigma2C = (r0_p[50]&r0_p[45])^(r0_p[20]&r0_p[10])^(r0_p[5]&r0_p[0])
        # compute σ2
        sigma2 = sigma2A ^ sigma2B ^ sigma2C
        # compute r1A
        temp = r0_p[40]^r0_p[35]^r0_p[30]^r0_p[25]^r0_p[15]^r0_p[0]
        R1A = sigma2 ^ temp
        # compute r1B
        R1B = ((r1_p[50]^r1_p[45])^(r1_p[40]^r1_p[20]))^((r1_p[10]^r1_p[5])^(r1_p[0]))
        r3_p = R1A^R1B
        z1 = (r1_p[0] + C_p[0]) % 2
        
        shift(r0_p,r_p,55)
        shift(r1_p,r3_p,55)
        shift(C_p,C_p[0],5)
        
        cap.append(z1)
        
    return np.array(cap) 

def genNavicCaCode_Pilot_Overlay(sv):
    """Build the PRN code for Pilot Overlay signal for a given satellite ID
    
    :param int sv: satellite code (1-64)
    :returns list: PRN  sequence for Overlay Pilot for chosen satellite
    
    """
    # init registers
#    G1 = [1 for i in range(10)]
    if(sv<1 or sv>64):
        print("Error: PRN ID out of bounds!")
        return None
    else:
        
        r0_pl = [int(i) for i in [*SV_L1_Pilot_Overlay_r0[sv]]]
        r1_pl = [int(i) for i in [*SV_L1_Pilot_Overlay_r1[sv]]]
        
    ca  = []# stuff piolt overlay output in here
    
    overlaycodeLength = 1800
        
    for l in range(overlaycodeLength):
        r_pl =  (r0_pl[5]^r0_pl[2])^(r0_pl[1]^r0_pl[0])
        #r_pl = shift(r0_pl, [6,3,2,1],[10])
        # compute σ2A
        sigma2A = (r0_pl[5]^r0_pl[2]) & (r0_pl[1]^r0_pl[0])
        # compute σ2B
        sigma2B = (r0_pl[5]&r0_pl[2]) ^ (r0_pl[1]&r0_pl[0])
        # compute σ2
        sigma2 = sigma2A ^ sigma2B 
        # compute r1A
        temp = r0_pl[6]^r0_pl[3]^r0_pl[2]^r0_pl[0]
        R1A = sigma2 ^ temp
        # compute r1B
        R1B = r1_pl[5]^r1_pl[2]^r1_pl[1]^r1_pl[0]
        r3_pl = R1A^R1B
        z2 = r1_pl[0]
        shift(r0_pl,r_pl,10)
        shift(r1_pl,r3_pl,10)      
    # modulo 2 add and append to the code
        ca.append(z2)    
   
    return np.array(ca)  
   

#function to upsample the Data PRN sequence generated to required sampling rate
def genNavicCaTable_Data(samplingFreq, codeLength, codeFreqBasis, satId ):
   # prnIdMax = 64
    #codeLength = 10230
    #codeFreqBasis = 1.023e6
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    #print(sampleCount)
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float32)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    #print(indexArr)
    return np.array([genNavicCaCode_Data(i) for i in satId])[:,indexArr].T

#function to upsample the Pilot PRN sequence generated to required sampling rate
def genNavicCaTable_Pilot(samplingFreq,codeLength, codeFreqBasis, satId):
    #prnIdMax = 64
    #codeLength = 10230
    #codeFreqBasis = 1.023e6
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float32)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    return np.array([genNavicCaCode_Pilot(i) for i in satId])[:,indexArr].T

#function to upsample the Pilot Overlay PRN sequence generated to required sampling rate
def genNavicCaTable_Pilot_Overlay(samplingFreq, codeLength, codeFreqBasis, satId):
  #  prnIdMax = 64
    #codeLength = 1800
    #codeFreqBasis = 1000
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float64)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    return np.array([genNavicCaCode_Pilot_Overlay(i) for i in satId])[:,indexArr].T

class NavicL1sModulator():
    def __init__(self, fs):
        self.sampleRate = fs
        self.codePhase = 0
        self.prnstart = 0
        self.cstart = 0

        # BOC(m,n) Init
        self.m1 = 1; self.n = 1
        self.m2 = 6
        fsc1 = self.m1*1.023e6
        epsilon1 = fsc1*1/(100*self.sampleRate)
        self.subCarrPhase1 = epsilon1 
        fsc2 = self.m2*1.023e6
        epsilon2 = fsc2*1/(100*self.sampleRate)
        self.subCarrPhase2 = epsilon2
        
    # columns of x have samples
    # columns of codeTable have sampled PRN sequence 
    def Modulate(self, x, codeTable1,codeTable2,codeTable3):

        codeNumSample_data = codeTable1.shape[0]
        codeNumSample_pilot = codeTable2.shape[0]
       
        
        numSample = x.shape[0]
        numChannel = x.shape[1]
        
        codeTable_p = codeTable2[(self.prnstart+np.arange(numSample))%codeNumSample_pilot]
        
        self.prnstart = ( self.prnstart+numSample)%codeNumSample_pilot
        
        # Subcarrier generation for BOC
        subCarr1Ch1 = self.__GenBocSubCarrier(numSample, self.m1)
        subCarr1Ch2 = self.__GenBocSubCarrier(numSample, self.m2)
        
        
        SubCarrSig1 = np.tile(np.array([subCarr1Ch1]).T, (1, numChannel))
        SubCarrSig2 = np.tile(np.array([subCarr1Ch2]).T, (1, numChannel))
        
        
        # PRN sequence of of pilot
        PilotCode = (codeTable_p+codeTable3)%2
        PilotSig = 1-2*PilotCode
        # Data 
        DataSig =1-2*np.logical_xor(x, codeTable1[np.arange(self.codePhase, self.codePhase+numSample)%codeNumSample_data, :])
        
        


        BocPilotSig1 = PilotSig * SubCarrSig1
        BocDataSig1 = DataSig * SubCarrSig1
        BocPilotsig6 = PilotSig * SubCarrSig2
    

        interplexSig = BocPilotSig1* BocDataSig1 *  BocPilotsig6

        self.codePhase = (self.codePhase+numSample)%codeNumSample_data

        alpha = (6/11)**0.5
        beta = (4/110)**0.5
        gamma = (4/11)**0.5
        eeta = (6/110)**0.5
        iqsig = (alpha*(BocPilotSig1)-beta* (BocPilotsig6 )) + 1j*(gamma*BocDataSig1+eeta*interplexSig)  # Document formula
        
        #iqsig = (alpha*(BocDataSig1)-beta* (interplexSig )) + 1j*(gamma*BocPilotSig1+eeta*BocPilotsig6)  # Document formula
        

        return iqsig
    
    def __GenBocSubCarrier(self, N, m):
        ts = 1/self.sampleRate
        t = np.arange(N)*ts
        fsc = m*1.023e6
         
        if(m == 1):
            #subCarrier = np.sign(np.sin(2*np.pi*(fsc*t + self.subCarrPhase1)))
            subCarrier = np.sign(np.sin(2*np.pi*fsc*t + self.subCarrPhase1))
            self.subCarrPhase1 += fsc*N*ts
            self.subCarrPhase1 -= int(self.subCarrPhase1)
        if(m == 6):
            #subCarrier = np.sign(np.sin(2*np.pi*(fsc*t + self.subCarrPhase2)))
            subCarrier = np.sign(np.sin(2*np.pi*fsc*t + self.subCarrPhase2))
            self.subCarrPhase2 += fsc*N*ts
            self.subCarrPhase2 -= int(self.subCarrPhase2)
        return subCarrier
    
    def Release(self,m):
        self.codePhase = 0

        fsc = m*1.023e6
        epsilon = fsc*1/(100*self.sampleRate)
        self.subCarrPhase = epsilon

#function to compute crc-24q parity
def rtk_crc24q(buff, length):
    """Build the CRC Code

    buff (contains the subframe bits right aligned i.e. additional zeros padded in first byte)\n
    For ex: If the bits are 1101010001111001010100100111 (28 bits), they are stored in buff as\n
    00001101 | 01000111 | 10010101 | 00100111\n
    b0   |    b1    |    b2    |    b3
    
    :param list buff: navdata
    :param int length: length of navdata
    :returns list crc: crc-24Q parity
    
    """    
    if length%8!=0:
        zero = 8-(length%8)
        buff = np.concatenate((np.zeros(zero,dtype=np.uint8),buff))
        length+=zero
    
    packed_data = np.packbits(buff.reshape(-1,8))
    crc = 0
    
    
    for i in range(int(length/8)):
        crc = ((crc << 8) & 0xFFFFFF) ^ tbl_CRC24Q[(crc >> 16) ^ packed_data[i]]

    #print(crc)
    return crc

tbl_CRC24Q = [
    0x000000,0x864CFB,0x8AD50D,0x0C99F6,0x93E6E1,0x15AA1A,0x1933EC,0x9F7F17,
    0xA18139,0x27CDC2,0x2B5434,0xAD18CF,0x3267D8,0xB42B23,0xB8B2D5,0x3EFE2E,
    0xC54E89,0x430272,0x4F9B84,0xC9D77F,0x56A868,0xD0E493,0xDC7D65,0x5A319E,
    0x64CFB0,0xE2834B,0xEE1ABD,0x685646,0xF72951,0x7165AA,0x7DFC5C,0xFBB0A7,
    0x0CD1E9,0x8A9D12,0x8604E4,0x00481F,0x9F3708,0x197BF3,0x15E205,0x93AEFE,
    0xAD50D0,0x2B1C2B,0x2785DD,0xA1C926,0x3EB631,0xB8FACA,0xB4633C,0x322FC7,
    0xC99F60,0x4FD39B,0x434A6D,0xC50696,0x5A7981,0xDC357A,0xD0AC8C,0x56E077,
    0x681E59,0xEE52A2,0xE2CB54,0x6487AF,0xFBF8B8,0x7DB443,0x712DB5,0xF7614E,
    0x19A3D2,0x9FEF29,0x9376DF,0x153A24,0x8A4533,0x0C09C8,0x00903E,0x86DCC5,
    0xB822EB,0x3E6E10,0x32F7E6,0xB4BB1D,0x2BC40A,0xAD88F1,0xA11107,0x275DFC,
    0xDCED5B,0x5AA1A0,0x563856,0xD074AD,0x4F0BBA,0xC94741,0xC5DEB7,0x43924C,
    0x7D6C62,0xFB2099,0xF7B96F,0x71F594,0xEE8A83,0x68C678,0x645F8E,0xE21375,
    0x15723B,0x933EC0,0x9FA736,0x19EBCD,0x8694DA,0x00D821,0x0C41D7,0x8A0D2C,
    0xB4F302,0x32BFF9,0x3E260F,0xB86AF4,0x2715E3,0xA15918,0xADC0EE,0x2B8C15,
    0xD03CB2,0x567049,0x5AE9BF,0xDCA544,0x43DA53,0xC596A8,0xC90F5E,0x4F43A5,
    0x71BD8B,0xF7F170,0xFB6886,0x7D247D,0xE25B6A,0x641791,0x688E67,0xEEC29C,
    0x3347A4,0xB50B5F,0xB992A9,0x3FDE52,0xA0A145,0x26EDBE,0x2A7448,0xAC38B3,
    0x92C69D,0x148A66,0x181390,0x9E5F6B,0x01207C,0x876C87,0x8BF571,0x0DB98A,
    0xF6092D,0x7045D6,0x7CDC20,0xFA90DB,0x65EFCC,0xE3A337,0xEF3AC1,0x69763A,
    0x578814,0xD1C4EF,0xDD5D19,0x5B11E2,0xC46EF5,0x42220E,0x4EBBF8,0xC8F703,
    0x3F964D,0xB9DAB6,0xB54340,0x330FBB,0xAC70AC,0x2A3C57,0x26A5A1,0xA0E95A,
    0x9E1774,0x185B8F,0x14C279,0x928E82,0x0DF195,0x8BBD6E,0x872498,0x016863,
    0xFAD8C4,0x7C943F,0x700DC9,0xF64132,0x693E25,0xEF72DE,0xE3EB28,0x65A7D3,
    0x5B59FD,0xDD1506,0xD18CF0,0x57C00B,0xC8BF1C,0x4EF3E7,0x426A11,0xC426EA,
    0x2AE476,0xACA88D,0xA0317B,0x267D80,0xB90297,0x3F4E6C,0x33D79A,0xB59B61,
    0x8B654F,0x0D29B4,0x01B042,0x87FCB9,0x1883AE,0x9ECF55,0x9256A3,0x141A58,
    0xEFAAFF,0x69E604,0x657FF2,0xE33309,0x7C4C1E,0xFA00E5,0xF69913,0x70D5E8,
    0x4E2BC6,0xC8673D,0xC4FECB,0x42B230,0xDDCD27,0x5B81DC,0x57182A,0xD154D1,
    0x26359F,0xA07964,0xACE092,0x2AAC69,0xB5D37E,0x339F85,0x3F0673,0xB94A88,
    0x87B4A6,0x01F85D,0x0D61AB,0x8B2D50,0x145247,0x921EBC,0x9E874A,0x18CBB1,
    0xE37B16,0x6537ED,0x69AE1B,0xEFE2E0,0x709DF7,0xF6D10C,0xFA48FA,0x7C0401,
    0x42FA2F,0xC4B6D4,0xC82F22,0x4E63D9,0xD11CCE,0x575035,0x5BC9C3,0xDD8538 ]
   


#class to generate navigation data at 100sps
class NavicDataGen():
    """NavicDataGen generates raw samples of data
    
    """
    
    def __init__(self, ds=100, fs=10*1.023e6, numChannel=1, file=None):
        """The init function is executed always when class is initiated

        :param int ds: data rate/bit rate
        :param int fs: sample rate
        :param int numChannel: number of Channels
        :param bool file: data file if present

        """
        self.dataRate = ds
        self.sampleRate = fs
        self.numSymbolsPerFrame = 1800
        self.numSymbolsPerSubFrame = 600
        self.numDataBitsPerSubFrame = 600
        #self.numDataBitsPerSubFrame = 300
        self.numSamplesPerBit = round(fs/ds)
        self.samplesToNextBit = self.numSamplesPerBit
        self.numChannel = numChannel
        self.bitcnt = 0

        self.symbolStream = np.empty((0, numChannel))
        self.dataStream = np.empty((0, numChannel))
        full_frame = np.empty(0)
        full_data = np.empty(0)
        for i in range(numChannel):
            encoded_data, unencoded_data = self.SymbolsGen()
            full_frame = np.append(full_frame,encoded_data)
            full_data = np.append(full_data,unencoded_data)

        self.symbolStream = np.append(self.symbolStream,full_frame.reshape(numChannel,-1).T,axis=0)
        self.dataStream = np.append(self.dataStream,full_data.reshape(numChannel,-1).T,axis=0)


    def GenerateBits(self, timeInterval):
      """Function to generate bits upto given time interval
      
      :param float timeInterval: time interval required
      :returns list genStream: bits upto given time interval
      
      """
      genStream = np.empty((1,self.numChannel))
      numBitsToGen = round(self.sampleRate*timeInterval)

      bufferCnt = numBitsToGen 
      # Main loop to generate sampled bits for given time interval
      while bufferCnt > 0: 
        # If remaining samples to generate is within the current bit's remaining duration
        if(bufferCnt < self.samplesToNextBit):
          # Copy current bit
          genStream = np.append(genStream, np.repeat(self.symbolStream[self.bitcnt: self.bitcnt+1,:], bufferCnt, axis=0), axis=0)
          # Update current bit's remaining duration
          self.samplesToNextBit -= bufferCnt
          # End loop
          bufferCnt = -1
        else:
          # Copy current bit for remaining duration
          genStream = np.append(genStream, np.repeat(self.symbolStream[self.bitcnt: self.bitcnt+1,:], self.samplesToNextBit, axis=0), axis=0)
          # Increment bit counter
          self.bitcnt+=1
          # If current frame ended, generate new frame
          if(self.bitcnt%self.numSymbolsPerFrame==0):
            full_frame = np.empty(0)
            full_data = np.empty(0)
            for i in range(self.numChannel):
                frame, data = self.SymbolsGen()
                full_frame = np.append(full_frame,frame)
                full_data = np.append(full_data,data)

            self.symbolStream = np.append(self.symbolStream,full_frame.reshape(self.numChannel,-1).T,axis=0)
            self.dataStream = np.append(self.dataStream,full_data.reshape(self.numChannel,-1).T,axis=0)
          # Update remaining samples to generate
          bufferCnt -= self.samplesToNextBit
          # Update remaining duration of current bit
          self.samplesToNextBit = self.numSamplesPerBit
      
      return genStream[1:numBitsToGen+1]
    
    def SymbolsGen(self):
        """Function to add CRC, tail bits, interleave and encode the data bits

        :returns list frame: encoded symbols 
        :returns list nav_data: pre-encoded data

        """
        nav_data = np.array([])
        frame = np.array([], dtype=int)
        frame_sub=np.array([], dtype=int)

        # Subframe 1 - BCH encoding (52 symbols)
        toi_data = randint(1,400)   
        toi_data_string=bin(toi_data)[2:].zfill(9)
        subframe1_data =  np.array([int(bit) for bit in toi_data_string])
        subframe1_encoded = self.bch_encode(subframe1_data)
        frame = np.append(frame, subframe1_encoded)
        nav_data=np.append(nav_data,subframe1_data)
        
        # Subframe 2 - LDPC encoding (1200 symbols)
        subframe2_data = np.array([np.random.randint(0, 2) for _ in range(576)])
        subframe2_crc = rtk_crc24q(subframe2_data, len(subframe2_data)) # CRC bits
        crc = "{:06X}".format(subframe2_crc)
        binary = bin(int(crc,16))[2:]
        padded_binary = list(binary.zfill(len(crc) * 4))
        nav_crc = np.append(subframe2_data,padded_binary)
        nav_crc = np.array([int(bit) for bit in nav_crc])
        A_s2,B_s2,C_s2,D_s2,E_s2,T_s2=self.Subframe2_SubMatrices()
        subframe2_encoded = self.ldpc_encode(A_s2,B_s2,C_s2,D_s2,E_s2,T_s2,nav_crc)
        frame_sub = np.append(frame_sub, subframe2_encoded)
        nav_data=np.append(nav_data,nav_crc)
    
        # Subframe 3 - LDPC encoding (548 symbols)
        subframe3_data = np.array([np.random.randint(0, 2) for _ in range(250)])
        subframe3_crc = rtk_crc24q(subframe3_data, len(subframe3_data))  # CRC bits
        crc = "{:06X}".format(subframe3_crc)
        binary = bin(int(crc,16))[2:]
        padded_binary = list(binary.zfill(len(crc) * 4))
        nav_crc = np.append(subframe3_data,padded_binary)
        nav_crc = np.array([int(bit) for bit in nav_crc])
        A_s3,B_s3,C_s3,D_s3,E_s3,T_s3=self.Subframe3_SubMatrices()
        subframe3_encoded = self.ldpc_encode(A_s3,B_s3,C_s3,D_s3,E_s3,T_s3,nav_crc)
        frame_sub = np.append(frame_sub, subframe3_encoded)
        nav_data=np.append(nav_data,nav_crc)
    
        #interleaving subframe2 and subframe3 bits (1748,)
        k=46
        n=38
        interleave = lambda x,k,n: x.reshape(n,-1).T.flatten()
        nav_intrlv = interleave(frame_sub,k,n)
        
        #Combining all the frames
        frame = np.append(frame, nav_intrlv)

        return frame, nav_data

    def Subframe2_SubMatrices(self):
        #Submatrix A 
        s="1,1 232,5 402,9 92,14 294,18 461,22 121,27 79,1 281,5 498,9 108,14 343,18 547,22 175,27 145,1 330,5 534,9 162,14 355,18 23,23 204,27 199,1 392,5 10,10 241,14 411,18 51,23 253,27 228,1 448,5 88,10 290,14 457,18 117,23 302,27 277,1 494,5 104,10 339,14 543,18 171,23 364,27 326,1 530,5 158,10 351,14 19,19 250,23 420,27 388,1 6,6 237,10 407,14 97,19 299,23 466,27 444,1 84,6 286,10 453,14 113,19 348,23 502,27 490,1 150,6 335,10 539,14 167,19 360,23 28,28 526,1 154,6 397,10 15,15 246,19 416,23 56,28 2,2 233,6 403,10 93,15 295,19 462,23 122,28 80,2 282,6 499,10 109,15 344,19 548,23 176,28 146,2 331,6 535,10 163,15 356,19 24,24 205,28 200,2 393,6 11,11 242,15 412,19 52,24 254,28 229,2 449,6 89,11 291,15 458,19 118,24 303,28 278,2 495,6 105,11 340,15 544,19 172,24 365,28 327,2 531,6 159,11 352,15 20,20 201,24 421,28 389,2 7,7 238,11 408,15 98,20 300,24 467,28 445,2 85,7 287,11 454,15 114,20 349,24 503,28 491,2 101,7 336,11 540,15 168,20 361,24 29,29 527,2 155,7 398,11 16,16 247,20 417,24 57,29 3,3 234,7 404,11 94,16 296,20 463,24 123,29 81,3 283,7 500,11 110,16 345,20 549,24 177,29 147,3 332,7 536,11 164,16 357,20 25,25 206,29 151,3 394,7 12,12 243,16 413,20 53,25 255,29 230,3 450,7 90,12 292,16 459,20 119,25 304,29 279,3 496,7 106,12 341,16 545,20 173,25 366,29 328,3 532,7 160,12 353,16 21,21 202,25 422,29 390,3 8,8 239,12 409,16 99,21 251,25 468,29 446,3 86,8 288,12 455,16 115,21 350,25 504,29 492,3 102,8 337,12 541,16 169,21 362,25 30,30 528,3 156,8 399,12 17,17 248,21 418,25 58,30 4,4 235,8 405,12 95,17 297,21 464,25 124,30 82,4 284,8 451,12 111,17 346,21 550,25 178,30 148,4 333,8 537,12 165,17 358,21 26,26 207,30 152,4 395,8 13,13 244,17 414,21 54,26 256,30 231,4 401,8 91,13 293,17 460,21 120,26 305,30 280,4 497,8 107,13 342,17 546,21 174,26 367,30 329,4 533,8 161,13 354,17 22,22 203,26 423,30 391,4 9,9 240,13 410,17 100,22 252,26 469,30 447,4 87,9 289,13 456,17 116,22 301,26 505,30 493,4 103,9 338,13 542,17 170,22 363,26 31,31 529,4 157,9 400,13 18,18 249,22 419,26 59,31 5,5 236,9 406,13 96,18 298,22 465,26 125,31 83,5 285,9 452,13 112,18 347,22 501,26 179,31 149,5 334,9 538,13 166,18 359,22 27,27 208,31 153,5 396,9 14,14 245,18 415,22 55,27 257,31 306,31 510,35 188,40 381,44 49,49 439,59 405,75 368,31 36,36 217,40 437,44 77,49 60,60 76,76 424,31 64,36 266,40 483,44 143,49 386,60 352,76 470,31 130,36 315,40 519,44 197,49 440,60 406,76 506,31 184,36 377,40 45,45 226,49 61,61 77,77 32,32 213,36 433,40 73,45 275,49 387,61 353,77 60,32 262,36 479,40 139,45 324,49 441,61 407,77 126,32 311,36 515,40 193,45 386,49 62,62 78,78 180,32 373,36 41,41 222,45 442,49 388,62 354,78 209,32 429,36 69,41 271,45 488,49 442,62 408,78 258,32 475,36 135,41 320,45 524,49 63,63 79,79 307,32 511,36 189,41 382,45 50,50 389,63 355,79 369,32 37,37 218,41 438,45 78,50 443,63 409,79 425,32 65,37 267,41 484,45 144,50 64,64 80,80 471,32 131,37 316,41 520,45 198,50 390,64 356,80 507,32 185,37 378,41 46,46 227,50 444,64 410,80 33,33 214,37 434,41 74,46 276,50 65,65 81,81 61,33 263,37 480,41 140,46 325,50 391,65 357,81 127,33 312,37 516,41 194,46 387,50 445,65 411,81 181,33 374,37 42,42 223,46 443,50 66,66 82,82 210,33 430,37 70,42 272,46 489,50 392,66 358,82 259,33 476,37 136,42 321,46 525,50 446,66 412,82 308,33 512,37 190,42 383,46 51,51 67,67 83,83 370,33 38,38 219,42 439,46 377,51 393,67 359,83 426,33 66,38 268,42 485,46 431,51 447,67 413,83 472,33 132,38 317,42 521,46 52,52 68,68 84,84 508,33 186,38 379,42 47,47 378,52 394,68 360,84 34,34 215,38 435,42 75,47 432,52 448,68 414,84 62,34 264,38 481,42 141,47 53,53 69,69 85,85 128,34 313,38 517,42 195,47 379,53 395,69 361,85 182,34 375,38 43,43 224,47 433,53 449,69 415,85 211,34 431,38 71,43 273,47 54,54 70,70 86,86 260,34 477,38 137,43 322,47 380,54 396,70 362,86 309,34 513,38 191,43 384,47 434,54 450,70 416,86 371,34 39,39 220,43 440,47 55,55 71,71 87,87 427,34 67,39 269,43 486,47 381,55 397,71 363,87 473,34 133,39 318,43 522,47 435,55 401,71 417,87 509,34 187,39 380,43 48,48 56,56 72,72 88,88 35,35 216,39 436,43 76,48 382,56 398,72 364,88 63,35 265,39 482,43 142,48 436,56 402,72 418,88 129,35 314,39 518,43 196,48 57,57 73,73 89,89 183,35 376,39 44,44 225,48 383,57 399,73 365,89 212,35 432,39 72,44 274,48 437,57 403,73 419,89 261,35 478,39 138,44 323,48 58,58 74,74 90,90 310,35 514,39 192,44 385,48 384,58 400,74 366,90 372,35 40,40 221,44 441,48 438,58 404,74 420,90 428,35 68,40 270,44 487,48 59,59 75,75 91,91 474,35 134,40 319,44 523,48 385,59 351,75 367,91 421,91 549,107 515,123 531,139 439,155 405,171 421,187 92,92 108,108 124,124 140,140 156,156 172,172 188,188 368,92 285,108 251,124 267,140 255,156 271,172 287,188 422,92 550,108 516,124 532,140 440,156 406,172 422,188 93,93 109,109 125,125 141,141 157,157 173,173 189,189 369,93 286,109 252,125 268,141 256,157 272,173 288,189 423,93 501,109 517,125 533,141 441,157 407,173 423,189 94,94 110,110 126,126 142,142 158,158 174,174 190,190 370,94 287,110 253,126 269,142 257,158 273,174 289,190 424,94 502,110 518,126 534,142 442,158 408,174 424,190 95,95 111,111 127,127 143,143 159,159 175,175 191,191 371,95 288,111 254,127 270,143 258,159 274,175 290,191 425,95 503,111 519,127 535,143 443,159 409,175 425,191 96,96 112,112 128,128 144,144 160,160 176,176 192,192 372,96 289,112 255,128 271,144 259,160 275,176 291,192 426,96 504,112 520,128 536,144 444,160 410,176 426,192 97,97 113,113 129,129 145,145 161,161 177,177 193,193 373,97 290,113 256,129 272,145 260,161 276,177 292,193 427,97 505,113 521,129 537,145 445,161 411,177 427,193 98,98 114,114 130,130 146,146 162,162 178,178 194,194 374,98 291,114 257,130 273,146 261,162 277,178 293,194 428,98 506,114 522,130 538,146 446,162 412,178 428,194 99,99 115,115 131,131 147,147 163,163 179,179 195,195 375,99 292,115 258,131 274,147 262,163 278,179 294,195 429,99 507,115 523,131 539,147 447,163 413,179 429,195 100,100 116,116 132,132 148,148 164,164 180,180 196,196 376,100 293,116 259,132 275,148 263,164 279,180 295,196 430,100 508,116 524,132 540,148 448,164 414,180 430,196 101,101 117,117 133,133 149,149 165,165 181,181 197,197 278,101 294,117 260,133 276,149 264,165 280,181 296,197 543,101 509,117 525,133 541,149 449,165 415,181 431,197 102,102 118,118 134,134 150,150 166,166 182,182 198,198 279,102 295,118 261,134 277,150 265,166 281,182 297,198 544,102 510,118 526,134 542,150 450,166 416,182 432,198 103,103 119,119 135,135 151,151 167,167 183,183 199,199 280,103 296,119 262,135 300,151 266,167 282,183 298,199 545,103 511,119 527,135 435,151 401,167 417,183 433,199 104,104 120,120 136,136 152,152 168,168 184,184 200,200 281,104 297,120 263,136 251,152 267,168 283,184 299,200 546,104 512,120 528,136 436,152 402,168 418,184 434,200 105,105 121,121 137,137 153,153 169,169 185,185 1,201 282,105 298,121 264,137 252,153 268,169 284,185 84,201 547,105 513,121 529,137 437,153 403,169 419,185 141,201 106,106 122,122 138,138 154,154 170,170 186,186 181,201 283,106 299,122 265,138 253,154 269,170 285,186 248,201 548,106 514,122 530,138 438,154 404,170 420,186 284,201 107,107 123,123 139,139 155,155 171,171 187,187 343,201 284,107 300,123 266,139 254,155 270,171 286,187 351,201 429,201 89,206 293,210 495,214 109,219 315,223 504,227 482,201 146,206 302,210 541,214 199,219 373,223 28,228 528,201 186,206 360,210 15,215 216,219 401,223 61,228 2,202 203,206 438,210 98,215 252,219 454,223 118,228 85,202 289,206 491,210 105,215 311,219 550,223 158,228 142,202 348,206 537,210 195,215 369,219 24,224 225,228 182,202 356,206 11,211 212,215 447,219 57,224 261,228 249,202 434,206 94,211 298,215 500,219 114,224 320,228 285,202 487,206 101,211 307,215 546,219 154,224 378,228 344,202 533,206 191,211 365,215 20,220 221,224 406,228 352,202 7,207 208,211 443,215 53,220 257,224 459,228 430,202 90,207 294,211 496,215 110,220 316,224 505,228 483,202 147,207 303,211 542,215 200,220 374,224 29,229 529,202 187,207 361,211 16,216 217,220 402,224 62,229 3,203 204,207 439,211 99,216 253,220 455,224 119,229 86,203 290,207 492,211 106,216 312,220 501,224 159,229 143,203 349,207 538,211 196,216 370,220 25,225 226,229 183,203 357,207 12,212 213,216 448,220 58,225 262,229 250,203 435,207 95,212 299,216 451,220 115,225 321,229 286,203 488,207 102,212 308,216 547,220 155,225 379,229 345,203 534,207 192,212 366,216 21,221 222,225 407,229 353,203 8,208 209,212 444,216 54,221 258,225 460,229 431,203 91,208 295,212 497,216 111,221 317,225 506,229 484,203 148,208 304,212 543,216 151,221 375,225 30,230 530,203 188,208 362,212 17,217 218,221 403,225 63,230 4,204 205,208 440,212 100,217 254,221 456,225 120,230 87,204 291,208 493,212 107,217 313,221 502,225 160,230 144,204 350,208 539,212 197,217 371,221 26,226 227,230 184,204 358,208 13,213 214,217 449,221 59,226 263,230 201,204 436,208 96,213 300,217 452,221 116,226 322,230 287,204 489,208 103,213 309,217 548,221 156,226 380,230 346,204 535,208 193,213 367,217 22,222 223,226 408,230 354,204 9,209 210,213 445,217 55,222 259,226 461,230 432,204 92,209 296,213 498,217 112,222 318,226 507,230 485,204 149,209 305,213 544,217 152,222 376,226 31,231 531,204 189,209 363,213 18,218 219,222 404,226 64,231 5,205 206,209 441,213 51,218 255,222 457,226 121,231 88,205 292,209 494,213 108,218 314,222 503,226 161,231 145,205 301,209 540,213 198,218 372,222 27,227 228,231 185,205 359,209 14,214 215,218 450,222 60,227 264,231 202,205 437,209 97,214 251,218 453,222 117,227 323,231 288,205 490,209 104,214 310,218 549,222 157,227 381,231 347,205 536,209 194,214 368,218 23,223 224,227 409,231 355,205 10,210 211,214 446,218 56,223 260,227 462,231 433,205 93,210 297,214 499,218 113,223 319,227 508,231 486,205 150,210 306,214 545,218 153,223 377,227 32,232 532,205 190,210 364,214 19,219 220,223 405,227 65,232 6,206 207,210 442,214 52,219 256,223 458,227 122,232 162,232 386,236 41,241 242,245 427,249 402,262 418,278 229,232 414,236 74,241 278,245 480,249 544,262 510,278 265,232 467,236 131,241 337,245 526,249 13,263 29,279 324,232 513,236 171,241 395,245 50,250 403,263 419,279 382,232 37,237 238,241 423,245 83,250 545,263 511,279 410,232 70,237 274,241 476,245 140,250 14,264 30,280 463,232 127,237 333,241 522,245 180,250 404,264 420,280 509,232 167,237 391,241 46,246 247,250 546,264 512,280 33,233 234,237 419,241 79,246 283,250 15,265 31,281 66,233 270,237 472,241 136,246 342,250 405,265 421,281 123,233 329,237 518,241 176,246 400,250 547,265 513,281 163,233 387,237 42,242 243,246 428,250 16,266 32,282 230,233 415,237 75,242 279,246 481,250 406,266 422,282 266,233 468,237 132,242 338,246 527,250 548,266 514,282 325,233 514,237 172,242 396,246 1,251 17,267 33,283 383,233 38,238 239,242 424,246 441,251 407,267 423,283 411,233 71,238 275,242 477,246 533,251 549,267 515,283 464,233 128,238 334,242 523,246 2,252 18,268 34,284 510,233 168,238 392,242 47,247 442,252 408,268 424,284 34,234 235,238 420,242 80,247 534,252 550,268 516,284 67,234 271,238 473,242 137,247 3,253 19,269 35,285 124,234 330,238 519,242 177,247 443,253 409,269 425,285 164,234 388,238 43,243 244,247 535,253 501,269 517,285 231,234 416,238 76,243 280,247 4,254 20,270 36,286 267,234 469,238 133,243 339,247 444,254 410,270 426,286 326,234 515,238 173,243 397,247 536,254 502,270 518,286 384,234 39,239 240,243 425,247 5,255 21,271 37,287 412,234 72,239 276,243 478,247 445,255 411,271 427,287 465,234 129,239 335,243 524,247 537,255 503,271 519,287 511,234 169,239 393,243 48,248 6,256 22,272 38,288 35,235 236,239 421,243 81,248 446,256 412,272 428,288 68,235 272,239 474,243 138,248 538,256 504,272 520,288 125,235 331,239 520,243 178,248 7,257 23,273 39,289 165,235 389,239 44,244 245,248 447,257 413,273 429,289 232,235 417,239 77,244 281,248 539,257 505,273 521,289 268,235 470,239 134,244 340,248 8,258 24,274 40,290 327,235 516,239 174,244 398,248 448,258 414,274 430,290 385,235 40,240 241,244 426,248 540,258 506,274 522,290 413,235 73,240 277,244 479,248 9,259 25,275 41,291 466,235 130,240 336,244 525,248 449,259 415,275 431,291 512,235 170,240 394,244 49,249 541,259 507,275 523,291 36,236 237,240 422,244 82,249 10,260 26,276 42,292 69,236 273,240 475,244 139,249 450,260 416,276 432,292 126,236 332,240 521,244 179,249 542,260 508,276 524,292 166,236 390,240 45,245 246,249 11,261 27,277 43,293 233,236 418,240 78,245 282,249 401,261 417,277 433,293 269,236 471,240 135,245 341,249 543,261 509,277 525,293 328,236 517,240 175,245 399,249 12,262 28,278 44,294 434,294 257,310 273,326 289,342 62,362 86,386 396,402 526,294 352,310 368,326 384,342 548,362 522,386 429,402 45,295 61,311 77,327 93,343 63,363 87,387 489,402 435,295 258,311 274,327 290,343 549,363 523,387 543,402 527,295 353,311 369,327 385,343 64,364 88,388 3,403 46,296 62,312 78,328 94,344 550,364 524,388 91,403 436,296 259,312 275,328 291,344 65,365 89,389 129,403 528,296 354,312 370,328 386,344 501,365 525,389 178,403 47,297 63,313 79,329 95,345 66,366 90,390 203,403 437,297 260,313 276,329 292,345 502,366 526,390 293,403 529,297 355,313 371,329 387,345 67,367 91,391 346,403 48,298 64,314 80,330 96,346 503,367 527,391 397,403 438,298 261,314 277,330 293,346 68,368 92,392 430,403 530,298 356,314 372,330 388,346 504,368 528,392 490,403 49,299 65,315 81,331 97,347 69,369 93,393 544,403 439,299 262,315 278,331 294,347 505,369 529,393 4,404 531,299 357,315 373,331 389,347 70,370 94,394 92,404 50,300 66,316 82,332 98,348 506,370 530,394 130,404 440,300 263,316 279,332 295,348 71,371 95,395 179,404 532,300 358,316 374,332 390,348 507,371 531,395 204,404 51,301 67,317 83,333 99,349 72,372 96,396 294,404 298,301 264,317 280,333 296,349 508,372 532,396 347,404 393,301 359,317 375,333 391,349 73,373 97,397 398,404 52,302 68,318 84,334 100,350 509,373 533,397 431,404 299,302 265,318 281,334 297,350 74,374 98,398 491,404 394,302 360,318 376,334 392,350 510,374 534,398 545,404 53,303 69,319 85,335 51,351 75,375 99,399 5,405 300,303 266,319 282,335 537,351 511,375 535,399 93,405 395,303 361,319 377,335 52,352 76,376 100,400 131,405 54,304 70,320 86,336 538,352 512,376 536,400 180,405 251,304 267,320 283,336 53,353 77,377 1,401 205,405 396,304 362,320 378,336 539,353 513,377 89,401 295,405 55,305 71,321 87,337 54,354 78,378 127,401 348,405 252,305 268,321 284,337 540,354 514,378 176,401 399,405 397,305 363,321 379,337 55,355 79,379 201,401 432,405 56,306 72,322 88,338 541,355 515,379 291,401 492,405 253,306 269,322 285,338 56,356 80,380 344,401 546,405 398,306 364,322 380,338 542,356 516,380 395,401 6,406 57,307 73,323 89,339 57,357 81,381 428,401 94,406 254,307 270,323 286,339 543,357 517,381 488,401 132,406 399,307 365,323 381,339 58,358 82,382 542,401 181,406 58,308 74,324 90,340 544,358 518,382 2,402 206,406 255,308 271,324 287,340 59,359 83,383 90,402 296,406 400,308 366,324 382,340 545,359 519,383 128,402 349,406 59,309 75,325 91,341 60,360 84,384 177,402 400,406 256,309 272,325 288,341 546,360 520,384 202,402 433,406 351,309 367,325 383,341 61,361 85,385 292,402 493,406 60,310 76,326 92,342 547,361 521,385 345,402 547,406 7,407 211,411 442,415 58,420 264,424 465,428 109,433 95,407 251,411 452,415 146,420 317,424 519,428 158,433 133,407 304,411 506,415 195,420 368,424 29,429 233,433 182,407 355,411 16,416 220,420 401,424 67,429 273,433 207,407 438,411 54,416 260,420 461,424 105,429 326,433 297,407 498,411 142,416 313,420 515,424 154,429 377,433 350,407 502,411 191,416 364,420 25,425 229,429 410,433 351,407 12,412 216,416 447,420 63,425 269,429 470,433 434,407 100,412 256,416 457,420 101,425 322,429 524,433 494,407 138,412 309,416 511,420 200,425 373,429 34,434 548,407 187,412 360,416 21,421 225,425 406,429 72,434 8,408 212,412 443,416 59,421 265,425 466,429 110,434 96,408 252,412 453,416 147,421 318,425 520,429 159,434 134,408 305,412 507,416 196,421 369,425 30,430 234,434 183,408 356,412 17,417 221,421 402,425 68,430 274,434 208,408 439,412 55,417 261,421 462,425 106,430 327,434 298,408 499,412 143,417 314,421 516,425 155,430 378,434 301,408 503,412 192,417 365,421 26,426 230,430 411,434 352,408 13,413 217,417 448,421 64,426 270,430 471,434 435,408 51,413 257,417 458,421 102,426 323,430 525,434 495,408 139,413 310,417 512,421 151,426 374,430 35,435 549,408 188,413 361,417 22,422 226,426 407,430 73,435 9,409 213,413 444,417 60,422 266,426 467,430 111,435 97,409 253,413 454,417 148,422 319,426 521,430 160,435 135,409 306,413 508,417 197,422 370,426 31,431 235,435 184,409 357,413 18,418 222,422 403,426 69,431 275,435 209,409 440,413 56,418 262,422 463,426 107,431 328,435 299,409 500,413 144,418 315,422 517,426 156,431 379,435 302,409 504,413 193,418 366,422 27,427 231,431 412,435 353,409 14,414 218,418 449,422 65,427 271,431 472,435 436,409 52,414 258,418 459,422 103,427 324,431 526,435 496,409 140,414 311,418 513,422 152,427 375,431 36,436 550,409 189,414 362,418 23,423 227,427 408,431 74,436 10,410 214,414 445,418 61,423 267,427 468,431 112,436 98,410 254,414 455,418 149,423 320,427 522,431 161,436 136,410 307,414 509,418 198,423 371,427 32,432 236,436 185,410 358,414 19,419 223,423 404,427 70,432 276,436 210,410 441,414 57,419 263,423 464,427 108,432 329,436 300,410 451,414 145,419 316,423 518,427 157,432 380,436 303,410 505,414 194,419 367,423 28,428 232,432 413,436 354,410 15,415 219,419 450,423 66,428 272,432 473,436 437,410 53,415 259,419 460,423 104,428 325,432 527,436 497,410 141,415 312,419 514,423 153,428 376,432 37,437 501,410 190,415 363,419 24,424 228,428 409,432 75,437 11,411 215,415 446,419 62,424 268,428 469,432 113,437 99,411 255,415 456,419 150,424 321,428 523,432 162,437 137,411 308,415 510,419 199,424 372,428 33,433 237,437 186,411 359,415 20,420 224,424 405,428 71,433 277,437 330,437 532,441 171,446 394,450 173,473 197,497 205,514 381,437 42,442 246,446 427,450 305,473 329,497 461,514 414,437 80,442 286,446 487,450 174,474 198,498 115,515 474,437 118,442 339,446 541,450 306,474 330,498 206,515 528,437 167,442 390,446 151,451 175,475 199,499 462,515 38,438 242,442 423,446 333,451 307,475 331,499 116,516 76,438 282,442 483,446 152,452 176,476 200,500 207,516 114,438 335,442 537,446 334,452 308,476 332,500 463,516 163,438 386,442 47,447 153,453 177,477 101,501 117,517 238,438 419,442 85,447 335,453 309,477 242,501 208,517 278,438 479,442 123,447 154,454 178,478 498,501 464,517 331,438 533,442 172,447 336,454 310,478 102,502 118,518 382,438 43,443 247,447 155,455 179,479 243,502 209,518 415,438 81,443 287,447 337,455 311,479 499,502 465,518 475,438 119,443 340,447 156,456 180,480 103,503 119,519 529,438 168,443 391,447 338,456 312,480 244,503 210,519 39,439 243,443 424,447 157,457 181,481 500,503 466,519 77,439 283,443 484,447 339,457 313,481 104,504 120,520 115,439 336,443 538,447 158,458 182,482 245,504 211,520 164,439 387,443 48,448 340,458 314,482 451,504 467,520 239,439 420,443 86,448 159,459 183,483 105,505 121,521 279,439 480,443 124,448 341,459 315,483 246,505 212,521 332,439 534,443 173,448 160,460 184,484 452,505 468,521 383,439 44,444 248,448 342,460 316,484 106,506 122,522 416,439 82,444 288,448 161,461 185,485 247,506 213,522 476,439 120,444 341,448 343,461 317,485 453,506 469,522 530,439 169,444 392,448 162,462 186,486 107,507 123,523 40,440 244,444 425,448 344,462 318,486 248,507 214,523 78,440 284,444 485,448 163,463 187,487 454,507 470,523 116,440 337,444 539,448 345,463 319,487 108,508 124,524 165,440 388,444 49,449 164,464 188,488 249,508 215,524 240,440 421,444 87,449 346,464 320,488 455,508 471,524 280,440 481,444 125,449 165,465 189,489 109,509 125,525 333,440 535,444 174,449 347,465 321,489 250,509 216,525 384,440 45,445 249,449 166,466 190,490 456,509 472,525 417,440 83,445 289,449 348,466 322,490 110,510 126,526 477,440 121,445 342,449 167,467 191,491 201,510 217,526 531,440 170,445 393,449 349,467 323,491 457,510 473,526 41,441 245,445 426,449 168,468 192,492 111,511 127,527 79,441 285,445 486,449 350,468 324,492 202,511 218,527 117,441 338,445 540,449 169,469 193,493 458,511 474,527 166,441 389,445 50,450 301,469 325,493 112,512 128,528 241,441 422,445 88,450 170,470 194,494 203,512 219,528 281,441 482,445 126,450 302,470 326,494 459,512 475,528 334,441 536,445 175,450 171,471 195,495 113,513 129,529 385,441 46,446 250,450 303,471 327,495 204,513 220,529 418,441 84,446 290,450 172,472 196,496 460,513 476,529 478,441 122,446 343,450 304,472 328,496 114,514 130,530 221,530 487,540 1,551 250,561 454,571 32,582 231,592 477,530 141,541 240,551 494,561 22,572 221,582 475,592 131,531 232,541 484,551 12,562 211,572 465,582 43,593 222,531 488,541 2,552 201,562 455,572 33,583 232,593 478,531 142,542 241,552 495,562 23,573 222,583 476,593 132,532 233,542 485,552 13,563 212,573 466,583 44,594 223,532 489,542 3,553 202,563 456,573 34,584 233,594 479,532 143,543 242,553 496,563 24,574 223,584 477,594 133,533 234,543 486,553 14,564 213,574 467,584 45,595 224,533 490,543 4,554 203,564 457,574 35,585 234,595 480,533 144,544 243,554 497,564 25,575 224,585 478,595 134,534 235,544 487,554 15,565 214,575 468,585 46,596 225,534 491,544 5,555 204,565 458,575 36,586 235,596 481,534 145,545 244,555 498,565 26,576 225,586 479,596 135,535 236,545 488,555 16,566 215,576 469,586 47,597 226,535 492,545 6,556 205,566 459,576 37,587 236,597 482,535 146,546 245,556 499,566 27,577 226,587 480,597 136,536 237,546 489,556 17,567 216,577 470,587 48,598 227,536 493,546 7,557 206,567 460,577 38,588 237,598 483,536 147,547 246,557 500,567 28,578 227,588 481,598 137,537 238,547 490,557 18,568 217,578 471,588 49,599 228,537 494,547 8,558 207,568 461,578 39,589 238,599 484,537 148,548 247,558 451,568 29,579 228,589 482,599 138,538 239,548 491,558 19,569 218,579 472,589 50,600 229,538 495,548 9,559 208,569 462,579 40,590 239,600 485,538 149,549 248,559 452,569 30,580 229,590 483,600 139,539 240,549 492,559 20,570 219,580 473,590 230,539 496,549 10,560 209,570 463,580 41,591 486,539 150,550 249,560 453,570 31,581 230,591 140,540 241,550 493,560 21,571 220,581 474,591 231,540 497,550 11,561 210,571 464,581 42,592"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((550,600))
        #print(a[0][0])
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'A' submatrix of size (550,600)
        A_s2=np.array(a)

        #Submatrix B
        s= "50,1 308,8 15,16 323,23 30,31 338,38 45,46 301,1 8,9 316,16 23,24 331,31 38,39 346,46 1,2 309,9 16,17 324,24 31,32 339,39 46,47 302,2 9,10 317,17 24,25 332,32 39,40 347,47 2,3 310,10 17,18 325,25 32,33 340,40 47,48 303,3 10,11 318,18 25,26 333,33 40,41 348,48 3,4 311,11 18,19 326,26 33,34 341,41 48,49 304,4 11,12 319,19 26,27 334,34 41,42 349,49 4,5 312,12 19,20 327,27 34,35 342,42 49,50 305,5 12,13 320,20 27,28 335,35 42,43 350,50 5,6 313,13 20,21 328,28 35,36 343,43 306,6 13,14 321,21 28,29 336,36 43,44 6,7 314,14 21,22 329,29 36,37 344,44 307,7 14,15 322,22 29,30 337,37 44,45 7,8 315,15 22,23 330,30 37,38 345,45"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((550,50))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'B' submatrix of size (550,50)
        B_s2=np.array(a)

        #Submatrix C
        s="48,1 34,37 7,223 7,359 43,395 6,431 12,467 49,2 35,38 8,224 8,360 44,396 7,432 13,468 50,3 36,39 9,225 9,361 45,397 8,433 14,469 1,4 37,40 10,226 10,362 46,398 9,434 15,470 2,5 38,41 11,227 11,363 47,399 10,435 16,471 3,6 39,42 12,228 12,364 48,400 11,436 17,472 4,7 40,43 13,229 13,365 26,401 12,437 18,473 5,8 41,44 14,230 14,366 27,402 13,438 19,474 6,9 42,45 15,231 15,367 28,403 14,439 20,475 7,10 43,46 16,232 16,368 29,404 15,440 21,476 8,11 44,47 17,233 17,369 30,405 16,441 22,477 9,12 45,48 18,234 18,370 31,406 17,442 23,478 10,13 46,49 19,235 19,371 32,407 18,443 24,479 11,14 47,50 20,236 20,372 33,408 19,444 25,480 12,15 35,201 21,237 21,373 34,409 20,445 26,481 13,16 36,202 22,238 22,374 35,410 21,446 27,482 14,17 37,203 23,239 23,375 36,411 22,447 28,483 15,18 38,204 24,240 24,376 37,412 23,448 29,484 16,19 39,205 25,241 25,377 38,413 24,449 30,485 17,20 40,206 26,242 26,378 39,414 25,450 31,486 18,21 41,207 27,243 27,379 40,415 46,451 32,487 19,22 42,208 28,244 28,380 41,416 47,452 33,488 20,23 43,209 29,245 29,381 42,417 48,453 34,489 21,24 44,210 30,246 30,382 43,418 49,454 35,490 22,25 45,211 31,247 31,383 44,419 50,455 36,491 23,26 46,212 32,248 32,384 45,420 1,456 37,492 24,27 47,213 33,249 33,385 46,421 2,457 38,493 25,28 48,214 34,250 34,386 47,422 3,458 39,494 26,29 49,215 49,351 35,387 48,423 4,459 40,495 27,30 50,216 50,352 36,388 49,424 5,460 41,496 28,31 1,217 1,353 37,389 50,425 6,461 42,497 29,32 2,218 2,354 38,390 1,426 7,462 43,498 30,33 3,219 3,355 39,391 2,427 8,463 44,499 31,34 4,220 4,356 40,392 3,428 9,464 45,500 32,35 5,221 5,357 41,393 4,429 10,465 33,36 6,222 6,358 42,394 5,430 11,466"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((50,600))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'C' submatrix of size (50,600)
        C_s2=np.array(a)

        #Submatrix D
        s="50,1 8,9 16,17 24,25 32,33 40,41 48,49 1,2 9,10 17,18 25,26 33,34 41,42 49,50 2,3 10,11 18,19 26,27 34,35 42,43 3,4 11,12 19,20 27,28 35,36 43,44 4,5 12,13 20,21 28,29 36,37 44,45 5,6 13,14 21,22 29,30 37,38 45,46 6,7 14,15 22,23 30,31 38,39 46,47 7,8 15,16 23,24 31,32 39,40 47,48"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((50,50))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'D' submatrix of shape (50,50)
        D_s2=np.array(a)

        #Submatrix E
        s="1,501 9,509 17,517 25,525 33,533 41,541 49,549 2,502 10,510 18,518 26,526 34,534 42,542 50,550 3,503 11,511 19,519 27,527 35,535 43,543 4,504 12,512 20,520 28,528 36,536 44,544 5,505 13,513 21,521 29,529 37,537 45,545 6,506 14,514 22,522 30,530 38,538 46,546 7,507 15,515 23,523 31,531 39,539 47,547 8,508 16,516 24,524 32,532 40,540 48,548"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((50,550))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'E' submatrix of shape (50,550)
        E_s2=np.array(a)

        #Submatrix T
        s="1,1 25,25 49,49 73,73 97,97 121,121 145,145 51,1 75,25 99,49 123,73 147,97 171,121 195,145 2,2 26,26 50,50 74,74 98,98 122,122 146,146 52,2 76,26 100,50 124,74 148,98 172,122 196,146 3,3 27,27 51,51 75,75 99,99 123,123 147,147 53,3 77,27 101,51 125,75 149,99 173,123 197,147 4,4 28,28 52,52 76,76 100,100 124,124 148,148 54,4 78,28 102,52 126,76 150,100 174,124 198,148 5,5 29,29 53,53 77,77 101,101 125,125 149,149 55,5 79,29 103,53 127,77 151,101 175,125 199,149 6,6 30,30 54,54 78,78 102,102 126,126 150,150 56,6 80,30 104,54 128,78 152,102 176,126 200,150 7,7 31,31 55,55 79,79 103,103 127,127 151,151 57,7 81,31 105,55 129,79 153,103 177,127 201,151 8,8 32,32 56,56 80,80 104,104 128,128 152,152 58,8 82,32 106,56 130,80 154,104 178,128 202,152 9,9 33,33 57,57 81,81 105,105 129,129 153,153 59,9 83,33 107,57 131,81 155,105 179,129 203,153 10,10 34,34 58,58 82,82 106,106 130,130 154,154 60,10 84,34 108,58 132,82 156,106 180,130 204,154 11,11 35,35 59,59 83,83 107,107 131,131 155,155 61,11 85,35 109,59 133,83 157,107 181,131 205,155 12,12 36,36 60,60 84,84 108,108 132,132 156,156 62,12 86,36 110,60 134,84 158,108 182,132 206,156 13,13 37,37 61,61 85,85 109,109 133,133 157,157 63,13 87,37 111,61 135,85 159,109 183,133 207,157 14,14 38,38 62,62 86,86 110,110 134,134 158,158 64,14 88,38 112,62 136,86 160,110 184,134 208,158 15,15 39,39 63,63 87,87 111,111 135,135 159,159 65,15 89,39 113,63 137,87 161,111 185,135 209,159 16,16 40,40 64,64 88,88 112,112 136,136 160,160 66,16 90,40 114,64 138,88 162,112 186,136 210,160 17,17 41,41 65,65 89,89 113,113 137,137 161,161 67,17 91,41 115,65 139,89 163,113 187,137 211,161 18,18 42,42 66,66 90,90 114,114 138,138 162,162 68,18 92,42 116,66 140,90 164,114 188,138 212,162 19,19 43,43 67,67 91,91 115,115 139,139 163,163 69,19 93,43 117,67 141,91 165,115 189,139 213,163 20,20 44,44 68,68 92,92 116,116 140,140 164,164 70,20 94,44 118,68 142,92 166,116 190,140 214,164 21,21 45,45 69,69 93,93 117,117 141,141 165,165 71,21 95,45 119,69 143,93 167,117 191,141 215,165 22,22 46,46 70,70 94,94 118,118 142,142 166,166 72,22 96,46 120,70 144,94 168,118 192,142 216,166 23,23 47,47 71,71 95,95 119,119 143,143 167,167 73,23 97,47 121,71 145,95 169,119 193,143 217,167 24,24 48,48 72,72 96,96 120,120 144,144 168,168 74,24 98,48 122,72 146,96 170,120 194,144 218,168 169,169 193,193 217,217 241,241 265,265 289,289 313,313 219,169 243,193 267,217 291,241 315,265 339,289 363,313 170,170 194,194 218,218 242,242 266,266 290,290 314,314 220,170 244,194 268,218 292,242 316,266 340,290 364,314 171,171 195,195 219,219 243,243 267,267 291,291 315,315 221,171 245,195 269,219 293,243 317,267 341,291 365,315 172,172 196,196 220,220 244,244 268,268 292,292 316,316 222,172 246,196 270,220 294,244 318,268 342,292 366,316 173,173 197,197 221,221 245,245 269,269 293,293 317,317 223,173 247,197 271,221 295,245 319,269 343,293 367,317 174,174 198,198 222,222 246,246 270,270 294,294 318,318 224,174 248,198 272,222 296,246 320,270 344,294 368,318 175,175 199,199 223,223 247,247 271,271 295,295 319,319 225,175 249,199 273,223 297,247 321,271 345,295 369,319 176,176 200,200 224,224 248,248 272,272 296,296 320,320 226,176 250,200 274,224 298,248 322,272 346,296 370,320 177,177 201,201 225,225 249,249 273,273 297,297 321,321 227,177 251,201 275,225 299,249 323,273 347,297 371,321 178,178 202,202 226,226 250,250 274,274 298,298 322,322 228,178 252,202 276,226 300,250 324,274 348,298 372,322 179,179 203,203 227,227 251,251 275,275 299,299 323,323 229,179 253,203 277,227 301,251 325,275 349,299 373,323 180,180 204,204 228,228 252,252 276,276 300,300 324,324 230,180 254,204 278,228 302,252 326,276 350,300 374,324 181,181 205,205 229,229 253,253 277,277 301,301 325,325 231,181 255,205 279,229 303,253 327,277 351,301 375,325 182,182 206,206 230,230 254,254 278,278 302,302 326,326 232,182 256,206 280,230 304,254 328,278 352,302 376,326 183,183 207,207 231,231 255,255 279,279 303,303 327,327 233,183 257,207 281,231 305,255 329,279 353,303 377,327 184,184 208,208 232,232 256,256 280,280 304,304 328,328 234,184 258,208 282,232 306,256 330,280 354,304 378,328 185,185 209,209 233,233 257,257 281,281 305,305 329,329 235,185 259,209 283,233 307,257 331,281 355,305 379,329 186,186 210,210 234,234 258,258 282,282 306,306 330,330 236,186 260,210 284,234 308,258 332,282 356,306 380,330 187,187 211,211 235,235 259,259 283,283 307,307 331,331 237,187 261,211 285,235 309,259 333,283 357,307 381,331 188,188 212,212 236,236 260,260 284,284 308,308 332,332 238,188 262,212 286,236 310,260 334,284 358,308 382,332 189,189 213,213 237,237 261,261 285,285 309,309 333,333 239,189 263,213 287,237 311,261 335,285 359,309 383,333 190,190 214,214 238,238 262,262 286,286 310,310 334,334 240,190 264,214 288,238 312,262 336,286 360,310 384,334 191,191 215,215 239,239 263,263 287,287 311,311 335,335 241,191 265,215 289,239 313,263 337,287 361,311 385,335 192,192 216,216 240,240 264,264 288,288 312,312 336,336 242,192 266,216 290,240 314,264 338,288 362,312 386,336 337,337 361,361 385,385 409,409 433,433 457,457 481,481 387,337 411,361 435,385 459,409 483,433 507,457 531,481 338,338 362,362 386,386 410,410 434,434 458,458 482,482 388,338 412,362 436,386 460,410 484,434 508,458 532,482 339,339 363,363 387,387 411,411 435,435 459,459 483,483 389,339 413,363 437,387 461,411 485,435 509,459 533,483 340,340 364,364 388,388 412,412 436,436 460,460 484,484 390,340 414,364 438,388 462,412 486,436 510,460 534,484 341,341 365,365 389,389 413,413 437,437 461,461 485,485 391,341 415,365 439,389 463,413 487,437 511,461 535,485 342,342 366,366 390,390 414,414 438,438 462,462 486,486 392,342 416,366 440,390 464,414 488,438 512,462 536,486 343,343 367,367 391,391 415,415 439,439 463,463 487,487 393,343 417,367 441,391 465,415 489,439 513,463 537,487 344,344 368,368 392,392 416,416 440,440 464,464 488,488 394,344 418,368 442,392 466,416 490,440 514,464 538,488 345,345 369,369 393,393 417,417 441,441 465,465 489,489 395,345 419,369 443,393 467,417 491,441 515,465 539,489 346,346 370,370 394,394 418,418 442,442 466,466 490,490 396,346 420,370 444,394 468,418 492,442 516,466 540,490 347,347 371,371 395,395 419,419 443,443 467,467 491,491 397,347 421,371 445,395 469,419 493,443 517,467 541,491 348,348 372,372 396,396 420,420 444,444 468,468 492,492 398,348 422,372 446,396 470,420 494,444 518,468 542,492 349,349 373,373 397,397 421,421 445,445 469,469 493,493 399,349 423,373 447,397 471,421 495,445 519,469 543,493 350,350 374,374 398,398 422,422 446,446 470,470 494,494 400,350 424,374 448,398 472,422 496,446 520,470 544,494 351,351 375,375 399,399 423,423 447,447 471,471 495,495 401,351 425,375 449,399 473,423 497,447 521,471 545,495 352,352 376,376 400,400 424,424 448,448 472,472 496,496 402,352 426,376 450,400 474,424 498,448 522,472 546,496 353,353 377,377 401,401 425,425 449,449 473,473 497,497 403,353 427,377 451,401 475,425 499,449 523,473 547,497 354,354 378,378 402,402 426,426 450,450 474,474 498,498 404,354 428,378 452,402 476,426 500,450 524,474 548,498 355,355 379,379 403,403 427,427 451,451 475,475 499,499 405,355 429,379 453,403 477,427 501,451 525,475 549,499 356,356 380,380 404,404 428,428 452,452 476,476 500,500 406,356 430,380 454,404 478,428 502,452 526,476 550,500 357,357 381,381 405,405 429,429 453,453 477,477 501,501 407,357 431,381 455,405 479,429 503,453 527,477 502,502 358,358 382,382 406,406 430,430 454,454 478,478 503,503 408,358 432,382 456,406 480,430 504,454 528,478 504,504 359,359 383,383 407,407 431,431 455,455 479,479 505,505 409,359 433,383 457,407 481,431 505,455 529,479 506,506 360,360 384,384 408,408 432,432 456,456 480,480 507,507 410,360 434,384 458,408 482,432 506,456 530,480 508,508 509,509 515,515 521,521 527,527 533,533 539,539 545,545 510,510 516,516 522,522 528,528 534,534 540,540 546,546 511,511 517,517 523,523 529,529 535,535 541,541 547,547 512,512 518,518 524,524 530,530 536,536 542,542 548,548 513,513 519,519 525,525 531,531 537,537 543,543 549,549 514,514 520,520 526,526 532,532 538,538 544,544 550,550"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((550,550))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'T' submatrix of shape (550,550)
        T_s2=np.array(a)
        return A_s2,B_s2,C_s2,D_s2,E_s2,T_s2

    def Subframe3_SubMatrices(self  ):
        #Submatrix A
        s="3,1 99,5 188,9 40,14 134,18 169,30 117,46 27,1 121,5 207,9 56,14 156,18 196,30 247,46 66,1 143,5 239,9 83,14 168,18 33,31 49,47 70,1 178,5 12,10 108,14 197,18 170,31 118,47 95,1 184,5 36,10 130,14 216,18 197,31 248,47 117,1 226,5 52,10 152,14 248,18 34,32 50,48 139,1 235,5 79,10 164,14 21,19 171,32 119,48 174,1 8,6 104,10 193,14 22,19 198,32 249,48 203,1 32,6 126,10 212,14 61,19 35,33 51,49 222,1 48,6 148,10 244,14 88,19 172,33 120,49 231,1 75,6 160,10 17,15 113,19 199,33 250,49 4,2 100,6 189,10 41,15 135,19 36,34 52,50 28,2 122,6 208,10 57,15 157,19 173,34 121,50 67,2 144,6 240,10 84,15 169,19 200,34 251,50 71,2 179,6 13,11 109,15 198,19 37,35 53,51 96,2 185,6 37,11 131,15 217,19 174,35 122,51 118,2 227,6 53,11 153,15 249,19 201,35 229,51 140,2 236,6 80,11 165,15 22,20 38,36 54,52 175,2 9,7 105,11 194,15 182,20 175,36 123,52 204,2 33,7 127,11 213,15 186,20 202,36 230,52 223,2 49,7 149,11 245,15 23,21 39,37 55,53 232,2 76,7 161,11 18,16 160,21 176,37 124,53 5,3 101,7 190,11 42,16 187,21 203,37 231,53 29,3 123,7 209,11 58,16 24,22 40,38 56,54 45,3 145,7 241,11 85,16 161,22 177,38 125,54 72,3 180,7 14,12 110,16 188,22 204,38 232,54 97,3 186,7 38,12 132,16 25,23 41,39 57,55 119,3 228,7 54,12 154,16 162,23 178,39 126,55 141,3 237,7 81,12 166,16 189,23 205,39 233,55 176,3 10,8 106,12 195,16 26,24 42,40 58,56 205,3 34,8 128,12 214,16 163,24 179,40 127,56 224,3 50,8 150,12 246,16 190,24 183,40 234,56 233,3 77,8 162,12 19,17 27,25 43,41 59,57 6,4 102,8 191,12 43,17 164,25 180,41 128,57 30,4 124,8 210,12 59,17 191,25 184,41 235,57 46,4 146,8 242,12 86,17 28,26 44,42 60,58 73,4 181,8 15,13 111,17 165,26 181,42 129,58 98,4 187,8 39,13 133,17 192,26 185,42 236,58 120,4 206,8 55,13 155,17 29,27 45,43 61,59 142,4 238,8 82,13 167,17 166,27 114,43 130,59 177,4 11,9 107,13 196,17 193,27 244,43 237,59 183,4 35,9 129,13 215,17 30,28 46,44 62,60 225,4 51,9 151,13 247,17 167,28 115,44 131,60 234,4 78,9 163,13 20,18 194,28 245,44 238,60 7,5 103,9 192,13 44,18 31,29 47,45 63,61 31,5 125,9 211,13 60,18 168,29 116,45 132,61 47,5 147,9 243,13 87,18 195,29 246,45 239,61 74,5 182,9 16,14 112,18 32,30 48,46 64,62  133,62 125,78 211,90 166,95 131,100 51,105 204,109 240,62 202,78 230,90 190,95 140,100 104,105 207,109 65,63 81,79 1,91 216,95 171,100 136,105 249,109 134,63 126,79 30,91 235,95 195,100 145,105 20,110 241,63 203,79 60,91 6,96 221,100 176,105 26,110 66,64 82,80 113,91 35,96 240,100 200,105 56,110 135,64 127,80 122,91 65,96 11,101 226,105 69,110 242,64 204,80 154,91 95,96 40,101 245,105 109,110 67,65 83,81 162,91 127,96 47,101 16,106 118,110 136,65 128,81 186,91 159,96 100,101 22,106 150,110 243,65 205,81 212,91 167,96 132,101 52,106 181,110 68,66 84,82 231,91 191,96 141,101 105,106 205,110 136,66 129,82 2,92 217,96 172,101 114,106 208,110 190,66 183,82 31,92 236,96 196,101 146,106 250,110 69,67 85,83 61,92 7,97 222,101 177,106 21,111 114,67 130,83 91,92 36,97 241,101 201,106 27,111 191,67 184,83 123,92 66,97 12,102 227,106 57,111 70,68 86,84 155,92 96,97 41,102 246,106 110,111 115,68 131,84 163,92 128,97 48,102 17,107 119,111 192,68 185,84 187,92 137,97 101,102 23,107 151,111 71,69 87,85 213,92 168,97 133,102 53,107 182,111 116,69 132,85 232,92 192,97 142,102 89,107 183,111 193,69 186,85 3,93 218,97 173,102 106,107 209,111 72,70 88,86 32,93 237,97 197,102 115,107 251,111 117,70 133,86 62,93 8,98 223,102 147,107 196,112 194,70 187,86 92,93 37,98 242,102 178,107 234,112 73,71 89,87 124,93 67,98 13,103 202,107 197,113 118,71 134,87 156,93 97,98 42,103 228,107 235,113 195,71 188,87 164,93 129,98 49,103 247,107 1,114 74,72 90,88 188,93 138,98 102,103 18,108 198,114 119,72 135,88 214,93 169,98 134,103 24,108 236,114 196,72 189,88 233,93 193,98 143,103 54,108 2,115 75,73 28,89 4,94 219,98 174,103 90,108 199,115 120,73 58,89 33,94 238,98 198,103 107,108 237,115 197,73 111,89 63,94 9,99 224,103 116,108 3,116 76,74 120,89 93,94 38,99 243,103 148,108 200,116 121,74 152,89 125,94 45,99 14,104 179,108 238,116 198,74 160,89 157,94 98,99 43,104 203,108 4,117 77,75 184,89 165,94 130,99 50,104 206,108 201,117 122,75 210,89 189,94 139,99 103,104 248,108 239,117 199,75 229,89 215,94 170,99 135,104 19,109 5,118 78,76 29,90 234,94 194,99 144,104 25,109 202,118 123,76 59,90 5,95 220,99 175,104 55,109 240,118 200,76 112,90 34,95 239,99 199,104 68,109 6,119 79,77 121,90 64,95 10,100 225,104 108,109 203,119 124,77 153,90 94,95 39,100 244,104 117,109 241,119 201,77 161,90 126,95 46,100 15,105 149,109 7,120 80,78 185,90 158,95 99,100 44,105 180,109 204,120 242,120 176,136 169,152 38,174 3,185 226,191 147,198 8,121 24,137 40,153 231,174 37,185 10,192 171,198 205,121 136,137 129,153 39,175 48,185 44,192 200,198 243,121 177,137 170,153 232,175 131,185 55,192 210,198 9,122 25,138 41,154 40,176 157,185 115,192 17,199 183,122 114,138 130,154 233,176 181,185 141,192 28,199 244,122 178,138 171,154 41,177 220,185 165,192 62,199 10,123 26,139 42,155 234,177 4,186 227,192 122,199 184,123 115,139 131,155 42,178 38,186 11,193 148,199 245,123 179,139 172,155 235,178 49,186 22,193 172,199 11,124 27,140 43,156 43,179 132,186 56,193 201,199 185,124 116,140 132,156 236,179 158,186 116,193 211,199 246,124 180,140 173,156 44,180 182,186 142,193 18,200 12,125 28,141 44,157 237,180 221,186 166,193 29,200 186,125 117,141 133,157 33,181 5,187 228,193 63,200 247,125 181,141 174,157 67,181 39,187 12,194 123,200 13,126 29,142 22,158 89,181 50,187 23,194 149,200 187,126 118,142 238,158 91,181 133,187 57,194 173,200 248,126 182,142 23,159 127,181 159,187 117,194 202,200 14,127 30,143 239,159 153,181 160,187 143,194 212,200 188,127 119,143 24,160 177,181 222,187 167,194 19,201 249,127 160,143 240,160 216,181 6,188 206,194 30,201 15,128 31,144 25,161 34,182 40,188 13,195 64,201 189,128 120,144 241,161 45,182 51,188 24,195 124,201 250,128 161,144 26,162 90,182 134,188 58,195 150,201 16,129 32,145 242,162 92,182 137,188 118,195 174,201 190,129 121,145 27,163 128,182 161,188 144,195 213,201 251,129 162,145 243,163 154,182 223,188 168,195 20,202 17,130 33,146 28,164 178,182 7,189 207,195 31,202 191,130 122,146 244,164 217,182 41,189 14,196 65,202 229,130 163,146 29,165 1,183 52,189 25,196 125,202 18,131 34,147 245,165 35,183 135,189 59,196 151,202 192,131 123,147 30,166 46,183 138,189 119,196 175,202 230,131 164,147 246,166 68,183 162,189 145,196 214,202 19,132 35,148 31,167 93,183 224,189 169,196 21,203 193,132 124,148 247,167 129,183 8,190 208,196 32,203 231,132 165,148 32,168 155,183 42,190 15,197 66,203 20,133 36,149 248,168 179,183 53,190 26,197 126,203 194,133 125,149 33,169 218,183 136,190 60,197 152,203 232,133 166,149 249,169 2,184 139,190 120,197 176,203 21,134 37,150 34,170 36,184 163,190 146,197 215,203 195,134 126,150 250,170 47,184 225,190 170,197 68,204 233,134 167,150 35,171 69,184 9,191 199,197 142,204 22,135 38,151 251,171 94,184 43,191 209,197 69,205 134,135 127,151 36,172 130,184 54,191 16,198 143,205 175,135 168,151 229,172 156,184 114,191 27,198 70,206 23,136 39,152 37,173 180,184 140,191 61,198 144,206 135,136 128,152 230,173 219,184 164,191 121,198 71,207 145,207 158,220 109,231 58,240 224,248 7,258 228,266 72,208 85,221 207,231 95,240 67,249 111,258 16,267 146,208 159,221 50,232 216,240 104,249 220,258 97,267 73,209 86,222 110,232 59,241 225,249 8,259 206,267 147,209 137,222 208,232 96,241 103,250 112,259 17,268 74,210 87,223 51,233 217,241 212,250 221,259 98,268 148,210 138,223 111,233 60,242 104,251 9,260 207,268 75,211 88,224 209,233 97,242 213,251 113,260 18,269 149,211 139,224 52,234 218,242 1,252 222,260 99,269 76,212 89,225 112,234 61,243 105,252 10,261 208,269 150,212 140,225 210,234 98,243 214,252 91,261 19,270 77,213 90,226 53,235 219,243 2,253 223,261 100,270 151,213 141,226 113,235 62,244 106,253 11,262 209,270 78,214 45,227 211,235 99,244 215,253 92,262 20,271 152,214 105,227 54,236 220,244 3,254 224,262 101,271 79,215 226,227 91,236 63,245 107,254 12,263 210,271 153,215 46,228 212,236 100,245 216,254 93,263 21,272 80,216 106,228 55,237 221,245 4,255 225,263 102,272 154,216 227,228 92,237 64,246 108,255 13,264 211,272 81,217 47,229 213,237 101,246 217,255 94,264 21,273 155,217 107,229 56,238 222,246 5,256 226,264 137,273 82,218 228,229 93,238 65,247 109,256 14,265 138,274 156,218 48,230 214,238 102,247 218,256 95,265 83,219 108,230 57,239 223,247 6,257 227,265 157,219 206,230 94,239 66,248 110,257 15,266 84,220 49,231 215,239 103,248 219,257 96,266"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((251,274))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'A' submatrix of size (251,274)
        A_s3=np.array(a)

        #Submatrix B
        s="139,1 4,5 146,8 11,12 153,15 18,19 23,23 1,2 143,5 8,9 150,12 15,16 157,19 140,2 5,6 147,9 12,13 154,16 19,20 2,3 144,6 9,10 151,13 16,17 158,20 141,3 6,7 148,10 13,14 155,17 20,21 3,4 145,7 10,11 152,14 17,18 159,21 142,4 7,8 149,11 14,15 156,18 22,22"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((251,23))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'B' submatrix of size (251,23)
        B_s3=np.array(a)

        #Submatrix C
        s="2,1 19,18 23,104 8,167 2,184 19,201 10,218 3,2 20,19 1,105 9,168 3,185 20,202 11,219 4,3 8,89 2,106 10,169 4,186 21,203 12,220 5,4 9,90 3,107 11,170 5,187 19,204 13,221 6,5 10,91 4,108 12,171 6,188 20,205 14,222 7,6 11,92 5,109 13,172 7,189 21,206 15,223 8,7 12,93 6,110 14,173 8,190 22,207 16,224 9,8 13,94 7,111 15,174 9,191 23,208 17,225 10,9 14,95 22,158 16,175 10,192 1,209 18,226 11,10 15,96 23,159 17,176 11,193 2,210 23,273 12,11 16,97 1,160 18,177 12,194 3,211 1,274 13,12 17,98 2,161 19,178 13,195 4,212 14,13 18,99 3,162 20,179 14,196 5,213 15,14 19,100 4,163 21,180 15,197 6,214 16,15 20,101 5,164 22,181 16,198 7,215 17,16 21,102 6,165 23,182 17,199 8,216 18,17 22,103 7,166 1,183 18,200 9,217"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((23,274))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'C' submatrix of size (23,274)
        C_s3=np.array(a)

        #Submatrix D
        s="21,20 22,21"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((23,23))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating 'D' submatrix of shape (23,23)
        D_s3=np.array(a)

        #Submatrix E
        s=" 1,229 5,233 9,237 13,241 17,245 21,249 2,230 6,234 10,238 14,242 18,246 22,250 3,231 7,235 11,239 15,243 19,247 23,251 4,232 8,236 12,240 16,244 20,248"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((23,251))
        for i in l:
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating E submatrix of shape (23,251)
        E_s3=np.array(a)

        #Submatrix T
        s="1,1 25,25 49,49 73,73 97,97 121,121 145,145 24,1 48,25 72,49 96,73 120,97 144,121 168,145 2,2 26,26 50,50 74,74 98,98 122,122 146,146 25,2 49,26 73,50 97,74 121,98 145,122 169,146 3,3 27,27 51,51 75,75 99,99 123,123 147,147 26,3 50,27 74,51 98,75 122,99 146,123 170,147 4,4 28,28 52,52 76,76 100,100 124,124 148,148 27,4 51,28 75,52 99,76 123,100 147,124 171,148 5,5 29,29 53,53 77,77 101,101 125,125 149,149 28,5 52,29 76,53 100,77 124,101 148,125 172,149 6,6 30,30 54,54 78,78 102,102 126,126 150,150 29,6 53,30 77,54 101,78 125,102 149,126 173,150 7,7 31,31 55,55 79,79 103,103 127,127 151,151 30,7 54,31 78,55 102,79 126,103 150,127 174,151 8,8 32,32 56,56 80,80 104,104 128,128 152,152 31,8 55,32 79,56 103,80 127,104 151,128 175,152 9,9 33,33 57,57 81,81 105,105 129,129 153,153 32,9 56,33 80,57 104,81 128,105 152,129 176,153 10,10 34,34 58,58 82,82 106,106 130,130 154,154 33,10 57,34 81,58 105,82 129,106 153,130 177,154 11,11 35,35 59,59 83,83 107,107 131,131 155,155 34,11 58,35 82,59 106,83 130,107 154,131 178,155 12,12 36,36 60,60 84,84 108,108 132,132 156,156 35,12 59,36 83,60 107,84 131,108 155,132 179,156 13,13 37,37 61,61 85,85 109,109 133,133 157,157 36,13 60,37 84,61 108,85 132,109 156,133 180,157 14,14 38,38 62,62 86,86 110,110 134,134 158,158 37,14 61,38 85,62 109,86 133,110 157,134 181,158 15,15 39,39 63,63 87,87 111,111 135,135 159,159 38,15 62,39 86,63 110,87 134,111 158,135 182,159 16,16 40,40 64,64 88,88 112,112 136,136 160,160 39,16 63,40 87,64 111,88 135,112 159,136 183,160 17,17 41,41 65,65 89,89 113,113 137,137 161,161 40,17 64,41 88,65 112,89 136,113 160,137 184,161 18,18 42,42 66,66 90,90 114,114 138,138 162,162 41,18 65,42 89,66 113,90 137,114 161,138 185,162 19,19 43,43 67,67 91,91 115,115 139,139 163,163 42,19 66,43 90,67 114,91 138,115 162,139 186,163 20,20 44,44 68,68 92,92 116,116 140,140 164,164 43,20 67,44 91,68 115,92 139,116 163,140 187,164 21,21 45,45 69,69 93,93 117,117 141,141 165,165 44,21 68,45 92,69 116,93 140,117 164,141 188,165 22,22 46,46 70,70 94,94 118,118 142,142 166,166 45,22 69,46 93,70 117,94 141,118 165,142 189,166 23,23 47,47 71,71 95,95 119,119 143,143 167,167 46,23 70,47 94,71 118,95 142,119 166,143 190,167 24,24 48,48 72,72 96,96 120,120 144,144 168,168 47,24 71,48 95,72 119,96 143,120 167,144 191,168 169,169 202,179 190,190 223,200 211,211 244,221 235,235 192,169 180,180 213,190 201,201 234,211 222,222 236,236 170,170 203,180 191,191 224,201 212,212 245,222 237,237 193,170 181,181 214,191 202,202 235,212 223,223 238,238 171,171 204,181 192,192 225,202 213,213 246,223 239,239 194,171 182,182 215,192 203,203 236,213 224,224 240,240 172,172 205,182 193,193 226,203 214,214 247,224 241,241 195,172 183,183 216,193 204,204 237,214 225,225 242,242 173,173 206,183 194,194 227,204 215,215 248,225 243,243 196,173 184,184 217,194 205,205 238,215 226,226 244,244 174,174 207,184 195,195 228,205 216,216 249,226 245,245 197,174 185,185 218,195 206,206 239,216 227,227 246,246 175,175 208,185 196,196 229,206 217,217 250,227 247,247 198,175 186,186 219,196 207,207 240,217 228,228 248,248 176,176 209,186 197,197 230,207 218,218 251,228 249,249 199,176 187,187 220,197 208,208 241,218 229,229 250,250 177,177 210,187 198,198 231,208 219,219 230,230 251,251 200,177 188,188 221,198 209,209 242,219 231,231 178,178 211,188 199,199 232,209 220,220 232,232 201,178 189,189 222,199 210,210 243,220 233,233 179,179 212,189 200,200 233,210 221,221 234,234"
        k=s.split()
        l=[]
        for i in k:
            s=i.split(",")
            l.append(s)
        a=np.zeros((251,251))
        for i in l: 
            a[int(i[0])-1][int(i[1])-1]=1
        #Creating T submatrix of shape (251,251)
        T_s3=np.array(a)
        return A_s3,B_s3,C_s3,D_s3,E_s3,T_s3
    
    def bch_encode(self, rawdata):
        # ----------------------Placeholder for BCH encoding logic
        encoded_data = []
        for i in range(52):
            encoded_data = np.append(rawdata[8],encoded_data) 
            encoded_data = encoded_data.astype(np.int16)
            fb = rawdata[0]^rawdata[1]^rawdata[3]^rawdata[4]^rawdata[5]^rawdata[6]^rawdata[7]^rawdata[8]
            rawdata = rawdata[:-1]
            rawdata = np.append(fb,rawdata)
        return encoded_data
    
    def ldpc_encode(self, A,B,C,D,E,T,S):
        # ---------------------------Placeholder for LDPC encoding logic
        pi=-(E@np.linalg.inv(T)@B)+D  #pi value of shape (50,50)
        p1_trans=-((np.linalg.inv(pi))@(-E@np.linalg.inv(T)@A+C)@S.T)%2  #creating p1_trans of shape (50,)
        p1=np.transpose(p1_trans)
        p2_trans=-((np.linalg.inv(T))@(A@S.T+B@p1_trans))%2 #creating p2_trans of shape (550,)
        p2=np.transpose(p2_trans)
        S_reshaped=S.reshape((len(S),1))
        p1_reshaped=p1.reshape((len(p1),1))
        p2_reshaped=p2.reshape((len(p2),1))
        combined_1d = np.concatenate((S_reshaped,p1_reshaped,p2_reshaped),axis=0)
        codeword= combined_1d.reshape(-1)
        return codeword
        
    def GetSymbolStream(self):
       """Function to return Symbolstream of symbols
             
       :returns list genStream: generated bits

       """
       numBitsGen = (self.bitcnt//self.numSymbolsPerSubFrame)*self.numDataBitsPerSubFrame
       remain = self.bitcnt%self.numSymbolsPerSubFrame
       if(remain> 0 ):
          numBitsGen += (remain)//2
       return self.symbolStream[0:numBitsGen]

    def GetDataStream(self):
        """Function to return Data Stream of navdata

        """
        return self.dataStream

#class to generate Pilot Overlay PRN Code at 100sps
class PilotOverlayBitGen():
    def __init__(self, prnId, codeLength, ds=100, fs=10*1.023e6):
      self.dataRate = ds
      self.sampleRate = fs
      self.numSamplesPerBit = round(fs/ds)
      self.samplesToNextBit = self.numSamplesPerBit
      self.pilotOverlayCodeLength = codeLength
      self.numChannel = len(prnId)
      self.pilotOverlayCode = np.array([genNavicCaCode_Pilot_Overlay(i) for i in prnId]).T
      self.bitStream = np.empty((1, self.numChannel))
      self.prnCount = 0
      self.bitStream[self.prnCount,:] = self.pilotOverlayCode[self.prnCount,:]
      self.prnCount = (self.prnCount+1)%self.pilotOverlayCodeLength

    def GenerateBits(self, timeInterval):
      genStream = np.empty((1,self.numChannel))
      numBitsToGen = round(self.sampleRate*timeInterval)

      bufferCnt = numBitsToGen
      while bufferCnt > 0:
        if(bufferCnt < self.samplesToNextBit):
          genStream = np.append(genStream, np.repeat(self.bitStream[-1:], bufferCnt, axis=0), axis=0)
          self.samplesToNextBit -= bufferCnt
          bufferCnt = -1
        else:
          genStream = np.append(genStream, np.repeat(self.bitStream[-1:], self.samplesToNextBit, axis=0), axis=0)
          self.bitStream = np.append(self.bitStream, [self.pilotOverlayCode[self.prnCount,:]], axis=0)
          self.prnCount = (self.prnCount+1)%self.pilotOverlayCodeLength
          bufferCnt -= self.samplesToNextBit
          self.samplesToNextBit = self.numSamplesPerBit
      
      return genStream[1:numBitsToGen+1]
    
    def GetBitStream(self):
       return self.bitStream 
 
      
# Channel model API
#the functions below simulate a channel, thereby create offsets and shift delays.
class PhaseFrequencyOffset():
  def __init__(self, sample_rate=1, phase_offset=0):
    self.phi = phase_offset
    self.dt = 1/sample_rate
    self.off_phi = 0

  def Offset(self, x, fShift):
    (N,M) = x.shape
    if(type(self.off_phi)==int):
      self.off_phi = np.zeros(M) 
    n = np.arange(0, N)
    arg = np.array([2*np.pi*n*fShift[i]*self.dt + self.off_phi[i] for i in range(0,M)]).T + self.phi
    self.off_phi += 2*np.pi*N*fShift*self.dt
    y = x * (np.cos(arg) + 1j*np.sin(arg))
    return y

  def Release(self):
    self.off_phi = 0

class IntegerDelay():
  def __init__(self, delays):
    self.D_buffer = [np.zeros(i) for i in delays.astype(int)]

  def Delay(self, x):
    y = np.zeros_like(x)
    N = x.shape[0]
    for i in range(0,len(self.D_buffer)):
      [y[:,i], self.D_buffer[i]] = np.split(np.append(self.D_buffer[i], x[:,i]), [N])
    return y


class FractionalDelay():
  def __init__(self, L=4, Dmax=100):
    self.L = L
    self.T = L-1
    if(Dmax > 65535):
      Dmax = 65535
    self.Dmax = Dmax
    self.Dmin = L//2 - 1
    self.H = np.linalg.inv(np.vander(np.arange(0,L), increasing=True).T)
    self.D_buffer = np.empty((0,0))
    self.Nch = -1

  def Delay(self, x, D):

    # If calling first time after empty delay buffer
    if(self.Nch < 0):
      self.Nch = D.shape[0]
      self.D_buffer = np.zeros((self.Dmax+self.T, self.Nch))
    elif(self.Nch != D.shape[0] or self.Nch != x.shape[1]):
      print("Error: Number of channels must remain constant between delay calls")
      return
  
    # Replace indexes with less/greater delay with Dmin/Dmax
    D[D < self.Dmin] = self.Dmin
    D[D > self.Dmax] = self.Dmax
    
    W = (D-self.Dmin).astype(int)
    f = self.Dmin+D-D.astype(int)
    # Columns of h contain filter coeffs
    h = self.H@np.array([f**i for i in range(0, self.L)])
    len = x.shape[0]

    temp = np.append(self.D_buffer, x, axis=0)
    self.D_buffer = temp[-self.D_buffer.shape[0]:]

    beg = self.D_buffer.shape[0]-W-self.T
    jump = len + self.T
    start = self.T
    end = self.T+len
    y = np.array([np.convolve(temp[beg[i]:beg[i]+jump, i], h[:,i])[start:end] for i in range(0,self.Nch)]).T

    return y

  def Release(self):
    self.Nch = -1
    self.D_buffer = np.empty((0,0))
    return


# Acquisition and Tracking API

def navic_pcps_acquisition(x, prnSeq_pilot, fs, fSearch,coherent_integ, non_coherent_integ, threshold=8):

    """Performs PCPS (Parallel Code Phase Search using FFT algorithm) acquisition

    :param x: Input signal buffer
    :param prnSeq_data: Sampled PRN sequence of data channel of satellite being searched
    :param prnSeq_pilot: Sampled PRN sequence of pilot channel of satellite being searched
    :param prnSeq_pilot_overlay: Sampled PRN sequence of pilot overlay of satellite being searched
    :param fs: Sampling rate
    :param fSearch: Array of Doppler frequencies to search
    :param threshold: Threshold value above which satellite is considered as visible/acquired, defaults to 0
    :return status, codeShift, dopplerShift: status is 'True' or 'False' for signal acquisition. In the case of staus being 'True', it provides coarse estimations of code phase and Doppler shift.
    """
    
    K = int(x.shape[0]/coherent_integ)
    ts = 1/fs
    t = np.arange(K)*ts
    
    
    
    #Multiply the pilot and overlay code sequence with Subcarrier
    fsc = 1.023e6  # Subcarrier1 - 1.023MHz
    epsilon = fsc*1/(100*fs)
    #subCarrier = np.sign(np.sin(2*np.pi*(fsc*t + epsilon)))
    subCarrier = np.sign(np.sin(2*np.pi*fsc*t + epsilon))
    
    
    
    pilotSig = 1-2*prnSeq_pilot
    
    prnSeq_pilot_FFT = np.conjugate(np.fft.fft(pilotSig*subCarrier))
    
   
    N = fSearch.shape[0]
   # ts = 1/fs
    #t = np.arange(K)*ts
    #start = 0 
    #end = K
    max_of_max = 0
    fDev = 0
    tau = 0
    for i in range(0,N):
        non_coherent_prod = np.zeros(K)
        Rxd = np.zeros(K)
        for  a in range(0,non_coherent_integ):
            coherent_prod = np.zeros(K, dtype = np.complex_)
            for b in range(0, coherent_integ):
                start = b*K
                end = (b*K)+(K)
                x_iq = x[start:end]*np.exp(-1j*2*np.pi*fSearch[i]*t)
                XFFT = np.fft.fft(x_iq)
                YFFT = XFFT*prnSeq_pilot_FFT
                coherent_prod = coherent_prod+YFFT
                                
            non_coherent_prod =  (1/K)*np.fft.ifft(coherent_prod)
            Rxd = Rxd + np.abs(non_coherent_prod)

        maxIndex = np.argmax(Rxd)
        max_val = Rxd[maxIndex]
    
        #Compute Peak to noise ratio
        for ind in range(-2,3,1):
            index = (maxIndex + ind+ K)%K
            Rxd[index] = 0
    
        noise = (np.sum(Rxd))/(K-5)
        p2n = max_val/noise
        
        if( p2n > max_of_max ) :
            max_of_max = p2n
            fDev = fSearch[i]
            tau = maxIndex
   
    '''
    plt.plot(np.abs(Rxd)**2)
    plt.ylim([0,0.05])
    plt.xlabel('time') ; plt.ylabel('Nav Data')
    l=np.abs(Rxd)**2
    time_values = np.arange(K)
    X, Y = np.meshgrid(time_values, fSearch)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('jet')  # You can use other colormaps as well
    norm = plt.Normalize(l.min(), l.max())
    colors = cmap(norm(l))
    surf = ax.plot_surface(X, Y, l.T, cmap=cmap)

    ax.set_xlabel('Time')
    ax.set_ylabel('Doppler Frequency')
    ax.set_zlabel('Rxd')
    ax.set_title('3D Plot of Matrix Data with Doppler Frequencies')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
    '''
    
    #powIn = np.mean(np.abs(x[0:K])**2)
    #sMax = np.abs(Rxd[maxRow, maxCol])**2
    #thresholdEst = 2*K*sMax/powIn
    
    

    #print("thresholdEst",thresholdEst)
    #print("powIn",powIn)
    #print("sMax",sMax)

    print("peak to noise ratio=", max_of_max) 
    if(max_of_max > threshold):
        return True, tau, fDev
    else:
        return False, 0, 0      

#This class is used to estimate SNR and Phase lock indicator
# https://gnss-sdr.org/docs/sp-blocks/tracking/#fn:Petovello10 : Implemented the Code Lock 
# Indicator and Phase lock Indicator as per the link

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
              estimated_sig_power = np.mean(np.abs(np.real(self.buf)))**2
              estimated_total_power = np.mean(np.abs(self.buf)**2)
              #print("sig power=", estimated_sig_power, "total=", estimated_total_power )
              rho_cap =  (estimated_sig_power)/(estimated_total_power - estimated_sig_power)
              CN0_cap = 10* math.log10(rho_cap) + 10*math.log10(self.sample_rate/2.0) - 10*math.log10(self.code_length)
              a = 10* math.log10(rho_cap)
              #print("rho=", rho_cap, "SNR=", a,"cn0=", CN0_cap)
              return CN0_cap      
        else : 
            return np.NAN
    def phase_lock_indicator(self) :
        l = len(self.buf)
        if (l == self.buf._capacity):
            a = np.sum(np.real(self.buf))**2
            b = np.sum(np.imag(self.buf))**2
            pli =  (a-b)/(a+b)
            return pli  
        else : 
            return np.NAN

#acquisition will provide rough frequency and code offsets. tracking will do precise calculation of frequency shifts and code delays
#thereby locks the values once threshold is reached
class NavicTracker:
    def __init__(self, prnId ):
        # Public, tunable properties
        self.InitialCodePhaseOffset = 0
        self.InitialDopplerShift = 0
        self.DisablePLL = False
        self.PLLIntegrationTime = 10  # In milliseconds
        

        # Signal properties
        self.PRNID = prnId
        self.CenterFrequency = 0
        self.SampleRate = 38.192e6  # In Hz
        

        # Properties of carrier tracking loops
        self.FLLOrder = 1
        self.PLLOrder = 2
        self.FLLNoiseBandwidth = 4
        self.PLLNoiseBandwidth = 18
        
        # Properties of code tracking loop
        self.DLLOrder = 1
        self.DLLNoiseBandwidth = 1

        # Pre-computed constants
        self.ChipRate = 1.023e6  # Chip rate of C/A-code

        # FLL properties
        self.pFLLNaturalFrequency = None
        self.pFLLGain1 = None
        self.pFLLGain2 = None
        self.pFLLGain3 = None
        self.pFLLWPrevious1 = 0
        self.pFLLWPrevious2 = 0
        self.pFLLNCOOut = 0
        
        # PLL properties
        self.pPLLNaturalFrequency = None
        self.pPLLGain1 = None
        self.pPLLGain2 = None
        self.pPLLGain3 = None
        self.pPLLWPrevious1 = 0
        self.pPLLWPrevious2 = 0
        self.pPLLNCOOut = 0
        self.pPreviousPhase = 0

        # DLL properties
        self.pDLLGain1 = None
        self.pDLLGain2 = None
        self.pDLLGain3 = None
        self.pDLLWPrevious1 = 0
        self.pDLLNCOOut = 0
        self.pDLLNaturalFrequency = None
        self.pPromptCode_pilot = None
        self.pPromptCode_data = None

        # General properties
        self.pNumIntegSamples = None
        self.pSamplesPerChip = None
        self.pReferenceCode = None
        self.pNumSamplesToAppend = 0
        self.pBuffer = None
        #self.pBuffer_data = None
        self.numSamplesPerCodeBlock = 0
        self.alpha = 0.6 # Pilot Discriminator weight
        
        #Code Tables
        
        
        self.dataCodeLength = 10230
        #self.pilotCodeLength = pilotCodeLength
        #self.pilotOverlayCodeLength = 10230
        #self.symbolRate = 100
        self.codeTable_data = genNavicCaCode_Data(self.PRNID).astype(float)
        self.codeTable_pilot = genNavicCaCode_Pilot(self.PRNID).astype(float)
        
        
        #Subcarrier
        self.subcarrierFrequency = 1.023e6 # Subcarrier frequency
        epsilon1 = self.subcarrierFrequency*1/(100*self.SampleRate)
        self.subCarrPhase1 = epsilon1 
        
        #SNR Calculation for Code lock indicator and Phase lock indicator
        self.bufsize_power_estimation = 10
        self.lock_fail_counter = 0
        self.lock_fail_counter_threshold = 50
        
                
    def updatePromptCode(self):
        # Initialize the code
        numCACodeBlocks = self.PLLIntegrationTime/10 # Each C/A-code block is of 10 milliseconds.
        
        ########################
        
       # codeNumSample_pilot = self.codeTable_pilot.shape[0]
        
        #overlay_rep = int(codeNumSample_pilot/self.PLLIntegrationTime)
        #symbols_covered = int(self.PLLIntegrationTime*1e-3*self.symbolRate)
        
        
        #codeTable_o = self.codeTable_pilot_overlay[self.cstart: (self.cstart+symbols_covered)]
        
        #codeTable_o_rep = np.repeat(codeTable_o, overlay_rep)
        
        #self.cstart = (self.cstart+symbols_covered)%self.pilotOverlayCodeLength
        
        
        
        # Subcarrier generation for BOC
        t = np.arange(int(self.SampleRate* self.PLLIntegrationTime*1e-3))/(int(self.SampleRate))
        #subCarrier = np.sign(np.sin(2*np.pi*(self.subcarrierFrequency*t+self.subCarrPhase1)))
        subCarrier = np.sign(np.sin(2*np.pi*self.subcarrierFrequency*t+self.subCarrPhase1))
        
        
        pilotSig = 1-2 * np.tile(self.__upsample_table(self.codeTable_pilot, self.SampleRate, len(self.codeTable_pilot)), numCACodeBlocks)
        
        dataCodeSig = 1-2*np.tile(self.__upsample_table(self.codeTable_data, self.SampleRate, len(self.codeTable_data)), numCACodeBlocks)

        
        
        self.pPromptCode_pilot = pilotSig*subCarrier      # added subcarrier multipication
        self.pPromptCode_data = dataCodeSig*subCarrier

        ######################
        self.pSamplesPerChip = self.SampleRate / self.ChipRate
        sampleFactor = Fraction(self.pSamplesPerChip)
        #upSampleFactor = sampleFactor.numerator; downSampleFactor = sampleFactor.denominator
        self.numSamplesPerCodeBlock = self.SampleRate * 10e-3 # As each code block is of 10e-3 seconds
        
        
    
    def setupImpl(self, dataCodeLength, U, cn0_min, detector_threshold, lock_fail_counter_threshold):

        # Perform one-time calculations, such as computing constants
        self.pNumIntegSamples = self.SampleRate*self.PLLIntegrationTime*1e-3
        # PLLIntegrationTime is in milliseconds. Hence multiply by 1e-3 to get it into sec

        # Calculate loop parameters for DLL
        if self.DLLOrder == 1: # Table 8.23 in Kaplan 3rd edition [1]
            self.pDLLNaturalFrequency = self.DLLNoiseBandwidth/0.25
            self.pDLLGain1 = self.pDLLNaturalFrequency
        elif self.DLLOrder == 2:
            self.pDLLNaturalFrequency = self.DLLNoiseBandwidth/0.53
            self.pDLLGain1 = self.pDLLNaturalFrequency**2
            self.pDLLGain2 = self.pDLLNaturalFrequency*1.414
        else: # self.DLLOrder == 3
            self.pDLLNaturalFrequency = self.DLLNoiseBandwidth/0.7845
            self.pDLLGain1 = self.pDLLNaturalFrequency**3
            self.pDLLGain2 = self.pDLLNaturalFrequency*1.1
            self.pDLLGain3 = self.pDLLNaturalFrequency*2.4

        # Calculate loop parameters for FLL
        if self.FLLOrder == 1: # Table 8.23 in Kaplan 3rd edition [1]
            self.pFLLNaturalFrequency = self.FLLNoiseBandwidth/0.25
            self.pFLLGain1 = self.pFLLNaturalFrequency
        elif self.FLLOrder == 2:
            self.pFLLNaturalFrequency = self.FLLNoiseBandwidth/0.53
            self.pFLLGain1 = self.pFLLNaturalFrequency**2
            self.pFLLGain2 = self.pFLLNaturalFrequency*1.414
        else: # self.FLLOrder == 3
            self.pFLLNaturalFrequency = self.FLLNoiseBandwidth/0.7845
            self.pFLLGain1 = self.pFLLNaturalFrequency**3
            self.pFLLGain2 = self.pFLLNaturalFrequency*1.1
            self.pFLLGain3 = self.pFLLNaturalFrequency*2.4

        # Calculate loop parameters for PLL
        if self.PLLOrder == 1: # Table 8.23 in Kaplan 3rd edition [1]
            self.pPLLNaturalFrequency = self.PLLNoiseBandwidth/0.25
            self.pPLLGain1 = self.pPLLNaturalFrequency
        elif self.PLLOrder == 2:
            self.pPLLNaturalFrequency = self.PLLNoiseBandwidth/0.53
            self.pPLLGain1 = self.pPLLNaturalFrequency**2
            self.pPLLGain2 = self.pPLLNaturalFrequency*1.414
        else: # self.PLLOrder == 3
            self.pPLLNaturalFrequency = self.PLLNoiseBandwidth/0.7845
            self.pPLLGain1 = self.pPLLNaturalFrequency**3
            self.pPLLGain2 = self.pPLLNaturalFrequency*1.1
            self.pPLLGain3 = self.pPLLNaturalFrequency*2.4
        
        #Lock Indicator parameters for Power Estimation and Phase Lock Indicator
        self.dataCodeLength = dataCodeLength
        self.bufsize_power_estimation = U
        self.lock_indicator = LockIndicator(self.bufsize_power_estimation,self.SampleRate , self.dataCodeLength)
        self.CN0_min = cn0_min 
        self.carrier_lock_detector_threshold = detector_threshold
        self.lock_fail_counter_threshold = lock_fail_counter_threshold
        self.updatePromptCode()
        
        # Calculate number of samples in delay
        numsamprot = round(self.InitialCodePhaseOffset * self.pSamplesPerChip) # Number of samples to rotate
        self.pNumSamplesToAppend = self.numSamplesPerCodeBlock - (numsamprot % self.numSamplesPerCodeBlock)
        
        
    def stepImpl(self, u):
        # Implement algorithm. Calculate y as a function of input u and
        # discrete states.
        
        
        
        coarsedelay = round(self.pNumSamplesToAppend)    # Me added round()
        numSamplesPerCodeBlock = self.SampleRate * 10e-3  # As each code block is of 1e-3 seconds
        finedelay = round(self.pDLLNCOOut * self.pSamplesPerChip)
        
        if len(self.pBuffer) != coarsedelay + finedelay:
            numextradelay = coarsedelay + finedelay - len(self.pBuffer)
            if numextradelay > 0:
                self.pBuffer = np.concatenate([np.zeros(numextradelay), self.pBuffer])
                #self.pBuffer_data = np.concatenate([np.zeros(numextradelay), self.pBuffer_data])
            else:  # numextradelay < 0. Equal to zero is not possible because of the first if condition
                if abs(numextradelay) < len(self.pBuffer):
                    # Remove samples from pBuffer itself
                    self.pBuffer = self.pBuffer[abs(numextradelay):]
                    #self.pBuffer_data = self.pBuffer_data[abs(numextradelay):]
                else:
                    n = numSamplesPerCodeBlock + numextradelay
                    self.pBuffer = np.concatenate([np.zeros(n), self.pBuffer])
                    #self.pBuffer_data = np.concatenate([np.zeros(n), self.pBuffer_data])
        
        #local_u = u
        # Buffer the input
        integtime = self.PLLIntegrationTime*1e-3 # PLLIntegrationTime is in milliseconds. Hence multiply by 1e-3 to get it into sec
       
        [u, self.pBuffer] = np.split(np.append(self.pBuffer, u), [round(self.SampleRate*integtime)])
        #[self.pBuffer,u] = np.split(np.append(self.pBuffer, u), [round(self.SampleRate*integtime)])
       
       #The below line is added to fix the bug related to receiveing correct data with one bit delay
        [localbuf, buf] = np.split(np.append(self.pBuffer, np.zeros(len(u))), [round(self.SampleRate*integtime)]) 
       
       
        # Carrier wipe-off
        #if (self.evensample_flag) : # Set frequency for even samples
        fc = self.CenterFrequency + self.InitialDopplerShift - self.pFLLNCOOut
            #self.previous_fc = self.CenterFrequency + self.InitialDopplerShift - self.pFLLNCOOut
            
        t = np.arange(self.pNumIntegSamples+1)/self.SampleRate
        #phases = (2*np.pi*fc*t + self.pPreviousPhase - self.pPLLNCOOut)
        phases = (2*np.pi*fc*t + self.pPreviousPhase - self.pPLLNCOOut)
        
        #iqsig = u * np.exp(-1j*phases[:-1])
        iqsig = localbuf * np.exp(-1j*phases[1:])
        #iqsig = localbuf * np.exp(-1j*phases[:-1])
        
        #GRAPH
        
        i=np.arange(-self.pSamplesPerChip,self.pSamplesPerChip,0.1)
        a_abs=[]
        '''for j in i:
            a=iqsig*np.roll(self.pPromptCode_pilot,round(j))
            a_sum=np.sum(a)
            a_abs.append(np.absolute(a_sum))
        a_abs=np.array(a_abs)
        plt.xlabel("I VALUES")
        plt.ylabel("Absolte Values")
        plt.plot(i,a_abs)
        plt.show()'''
        
        self.pPreviousPhase = (phases[-1] + self.pPLLNCOOut)
        
        numSamplesPerHalfChip = round(self.pSamplesPerChip*0.15)
        numSamplesPerThreeQuarterChip = round(self.pSamplesPerChip*0.6)
        #numSamplesPerHalfChip = round(self.pSamplesPerChip*(-0.35))
        #numSamplesPerThreeQuarterChip = round(self.pSamplesPerChip*0.1)
        # Code wipe-off Pilot
        # Update the prompt code appropriately
        iq_e_pilot = iqsig*np.roll(self.pPromptCode_pilot, -1*numSamplesPerHalfChip) # Early
        iq_ve_pilot =iqsig*np.roll(self.pPromptCode_pilot, -1*numSamplesPerThreeQuarterChip)# Very Early
        iq_p_pilot = iqsig*self.pPromptCode_pilot # Prompt for Pilot
        iq_l_pilot = iqsig*np.roll(self.pPromptCode_pilot, numSamplesPerHalfChip)# Late
        iq_vl_pilot =iqsig*np.roll(self.pPromptCode_pilot, numSamplesPerThreeQuarterChip) # Very Late
        
        integeval_pilot = np.mean(iq_e_pilot)
        integlval_pilot = np.mean(iq_l_pilot)
        integveval_pilot = np.mean(iq_ve_pilot)
        integvlval_pilot = np.mean(iq_vl_pilot)
        integpval_pilot=np.mean(iq_p_pilot)
        #print("pilot_prompt",np.abs(integpval_pilot))
        #print("pilot_early_val:",np.abs(integeval_pilot))
        #print("pilot_late_val:",np.abs(integlval_pilot))
        #print("pilot_ve_val:",np.abs(integveval_pilot))
        #print("pilot_vl_val:",np.abs(integvlval_pilot))
        
        #Code wipe off for Data Channel
        iq_e_data = iqsig*np.roll(self.pPromptCode_data, -1*numSamplesPerHalfChip) # Early
        iq_ve_data = iqsig*np.roll(self.pPromptCode_data, -1*numSamplesPerThreeQuarterChip)# Very Early
        iq_p_data = iqsig*self.pPromptCode_data # Prompt for Data channel
        iq_l_data = iqsig*np.roll(self.pPromptCode_data, numSamplesPerHalfChip) # Late
        iq_vl_data =iqsig*np.roll(self.pPromptCode_data, numSamplesPerThreeQuarterChip) # Very Late
        
        integeval_data = np.mean(iq_e_data)
        integlval_data = np.mean(iq_l_data)
        integveval_data = np.mean(iq_ve_data)
        integvlval_data = np.mean(iq_vl_data)
        integpval_data=np.mean(iq_p_data)
       # print("data_prompt",np.abs(integpval_data))
        #print("data_early_val:",np.abs(integeval_data))
        #print("data_late_val:",np.abs(integlval_data))
        #print("data_ve_val:",np.abs(integveval_data))
        #print("data_vl_val:",np.abs(integvlval_data))

        #Extract Data for Pilot Channel
        millisecdata_pilot = iq_p_pilot.reshape((int(self.PLLIntegrationTime/10), -1)).T # Each column contains ten millisecond of data 
        y_pilot = np.sum(millisecdata_pilot, axis=0) # Each element contains integrated value of one millisecond of data
        #y_pilot = np.mean(millisecdata_pilot, axis=0)
        integpval_pilot = np.sum(y_pilot)
        #print("prompt",integpval_pilot)
        
        if len(iq_p_pilot) % 2 != 0: # Odd number of samples
            fllin_pilot = np.mean(np.reshape(np.concatenate([iq_p_pilot, [0]]), (2, -1)).T, axis=0) # Append a zero
            #fllin_pilot = np.mean(np.reshape(np.concatenate([remove_overlay, [0]]), (2, -1)).T, axis=0)
        else:
            fllin_pilot = np.mean(iq_p_pilot.reshape((2, -1)).T, axis=0)
            #fllin_pilot = np.mean(remove_overlay.reshape((2, -1)).T, axis=0)
        
        
        #Extract Data from Data Channel
        millisecdata_data = iq_p_data.reshape((int(self.PLLIntegrationTime/10), -1)).T # Each column contains ten millisecond of data
        
        y_data = np.sum(millisecdata_data, axis=0) # Each element contains integrated value of one millisecond of data
        #y_data = np.mean(millisecdata_data, axis=0) 
        integpval_data = np.sum(y_data)
        
        
        if len(iq_p_data) % 2 != 0: # Odd number of samples
            fllin_data = np.mean(np.reshape(np.concatenate([iq_p_data, [0]]), (2, -1)).T, axis=0) # Append a zero
            
        else:
            fllin_data = np.mean(iq_p_data.reshape((2, -1)).T, axis=0)
            
        
        # DLL discriminator Pilot
        E_pilot = np.linalg.norm([integeval_pilot, integveval_pilot])
        L_pilot = np.linalg.norm([integlval_pilot, integvlval_pilot])
        
        delayerr_pilot = 0 
        #delayerr = (E-L)/(2*(E+L)) # Non-coherent early minus late normalized detector
        if ( (E_pilot + L_pilot ) != 0) :
           delayerr_pilot = (E_pilot - L_pilot )/(2*(E_pilot + L_pilot )) # Non-coherent early minus late normalized detector
        
        # DLL discriminator Data
        E_data = np.linalg.norm([integeval_data, integveval_data])
        L_data = np.linalg.norm([integlval_data, integvlval_data])
        
        delayerr_data = 0 
        #delayerr = (E-L)/(2*(E+L)) # Non-coherent early minus late normalized detector
        if ( (E_data + L_data) != 0) :
           delayerr_data = (E_data-L_data)/(2*(E_data + L_data)) # Non-coherent early minus late normalized detector
        
        delayerr = self.alpha * delayerr_pilot + (1-self.alpha)* delayerr_data
        #delayerr = delayerr_pilot
        
        # DLL loop filter
        if self.DLLOrder == 2:
            # 1st integrator
            wcurrent = delayerr*self.pDLLGain1*integtime + self.pDLLWPrevious1
            loopfilterout = (wcurrent + self.pDLLWPrevious1)/2 + delayerr*self.pDLLGain2
            self.pDLLWPrevious1 = wcurrent  # Acceleration accumulator
        elif self.DLLOrder == 1:
            loopfilterout = delayerr*self.pDLLGain1

        # DLL NCO
        delaynco = self.pDLLNCOOut+integtime*loopfilterout
        #delaynco=integtime*loopfilterout
        self.pDLLNCOOut = delaynco
        
        
        
        pherr_pilot = 0
        pherr_data = 0
        # PLL discriminator
        if self.DisablePLL:
            pherr_pilot = 0
            pherr_data = 0
        else:
            #pherr = np.arctan(np.real(integpval)/np.imag(integpval))
            #pherr = np.arctan2(np.real(integpval),np.imag(integpval))/(2*np.pi)
            if (np.imag(integpval_pilot) != 0 ) :
                pherr_pilot = np.arctan(np.real(integpval_pilot)/np.imag(integpval_pilot))
                #pherr_pilot = np.arctan2(np.real(integpval_pilot),np.imag(integpval_pilot))
            if (np.real(integpval_data) != 0 ) :
                pherr_data = np.arctan(np.imag(integpval_data)/np.real(integpval_data))
        
        #pherr = self.alpha * pherr_pilot + (1-self.alpha)* pherr_data
        pherr = pherr_pilot
        
        # PLL loop filter
        if self.PLLOrder == 3:
            # 1st integrator
            wcurrent = pherr*self.pPLLGain1*integtime + self.pPLLWPrevious1
            integ1out = (wcurrent + self.pPLLWPrevious1)/2 + pherr*self.pPLLGain2
            self.pPLLWPrevious1 = wcurrent # Acceleration accumulator

            # 2nd integrator
            wcurrent = integ1out*integtime + self.pPLLWPrevious2
            loopfilterout = (wcurrent + self.pPLLWPrevious2)/2 + pherr*self.pPLLGain3
            self.pPLLWPrevious2 = wcurrent # Velocity accumulator
        elif self.PLLOrder == 2:
            wcurrent = pherr*self.pPLLGain1*integtime + self.pPLLWPrevious1
            loopfilterout = (wcurrent + self.pPLLWPrevious1)/2 + pherr*self.pPLLGain2
            self.pPLLWPrevious1 = wcurrent # Velocity accumulator

        # PLL NCO
        phnco = self.pPLLNCOOut + integtime*loopfilterout
        #phnco=integtime*loopfilterout
        self.pPLLNCOOut = phnco
        
        fqyerr = 0
        fqyerr_pilot =0
        fqyerr_data = 0
        
        # FLL discriminator Pilot
        phasor_pilot = np.conj(fllin_pilot[0])*fllin_pilot[1]
        
        fqyerr_pilot =  -np.angle(phasor_pilot)/(np.pi*integtime)
        
        
        # FLL discriminator Data
        phasor_data = np.conj(fllin_data[0])*fllin_data[1]
        
        fqyerr_data = -np.angle(phasor_data)/(np.pi*integtime)
        
        '''
        
        if (self.evensample_flag) :
            self.previous_pherr_pilot = pherr_pilot
            self.previous_pherr_data = pherr_data
        else :      # Compute Frequency error using Differential Arctangent Discriminator method
            val = self.__phase_for_fqyerr(pherr_pilot, self.previous_pherr_pilot)
            fqyerr_pilot = val/(2*np.pi*integtime)
            val = self.__phase_for_fqyerr(pherr_data, self.previous_pherr_data)
            fqyerr_data = val/(2*np.pi*integtime)
        '''
        
        fqyerr = self.alpha * fqyerr_pilot + (1-self.alpha)* fqyerr_data
        
        #fqyerr = fqyerr_pilot
        
        
        # FLL loop filter
        if self.FLLOrder == 2:
        # 1st integrator
            wcurrent = fqyerr*self.pFLLGain1*integtime + self.pFLLWPrevious1
            loopfilterout = (wcurrent + self.pFLLWPrevious1)/2 + fqyerr*self.pFLLGain2
            self.pFLLWPrevious1 = wcurrent # Acceleration accumulator
        elif self.FLLOrder == 1:
            loopfilterout = fqyerr*self.pFLLGain1

        # FLL NCO
        fqynco = self.pFLLNCOOut + integtime*loopfilterout
        #fqynco=integtime*loopfilterout
        self.pFLLNCOOut = fqynco

        #self.updatePromptCode() ##Update the prompt code for next set of sample
        # Phase lock indictor and SNR Threshold calcuator
        cn0_cap, pli = self._Lock_indicator_calc(y_data)
        
        if ( (not math.isnan(cn0_cap))  and (not math.isnan(pli)) ) :
            if ( cn0_cap <  self.CN0_min or pli < self.carrier_lock_detector_threshold ) :
                self.lock_fail_counter  += 1
            else :
                 self.lock_fail_counter = max(self.lock_fail_counter-1, 0)
        else :
            self.lock_fail_counter  += 1
        
        return y_pilot, y_data, fqyerr, fqynco, pherr, phnco, delayerr, delaynco, cn0_cap, pli, self.lock_fail_counter,fc
        #return y_pilot, y_data, fqyerr, self.pFLLNCOOut, pherr, phnco, delayerr, delaynco,self.previous_fc
    
    def _Lock_indicator_calc(self, y_data) :
        
        self.lock_indicator.addVal (y_data)
        return self.lock_indicator.CN0_cap(), self.lock_indicator.phase_lock_indicator()
    

    def __upsample_table(self, codeBase, samplingFreq, codeLength ):
        """Upsample PRN sequence of satellite being tracked
     
        :param list codeBase: PRN sequence for complete period
        :param list samplingFreq: Desired sampling frequency
        :returns list y: Sampled PRN sequence"""
        #codeLength = 1023
        codeFreqBasis = self.ChipRate
        samplingPeriod = 1/samplingFreq
        sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
        indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float32)     # Avoid floating point error due to high precision
        indexArr = indexArr.astype(int)
        return codeBase[indexArr]
    
    def resetImpl(self):
        # Initialize / reset discrete-state properties
        self.pBuffer = np.zeros(round(self.pNumSamplesToAppend))
        self.pFLLWPrevious1 = 0
        self.pFLLWPrevious2 = 0
        self.pFLLNCOOut = 0
        self.pPLLWPrevious1 = 0
        self.pPLLWPrevious2 = 0
        self.pPLLNCOOut = 0
        self.pDLLWPrevious1 = 0
        self.pDLLNCOOut = 0
#end of acquisition and tracking code


class decoder():
    """Function decodes the subframes
        Subframe1 - BCH Decoding
        Subframe2 and Subframe3 - LDPC Decoding
    """

    def __init__(self):
        self.instance=NavicDataGen()
        A_s2,B_s2,C_s2,D_s2,E_s2,T_s2=self.instance.Subframe2_SubMatrices()
        A_s3,B_s3,C_s3,D_s3,E_s3,T_s3=self.instance.Subframe3_SubMatrices()
        self.A_s2=A_s2
        self.B_s2=B_s2
        self.C_s2=C_s2
        self.D_s2=D_s2
        self.E_s2=E_s2
        self.T_s2=T_s2
        self.A_s3=A_s3
        self.B_s3=B_s3
        self.C_s3=C_s3
        self.D_s3=D_s3
        self.E_s3=E_s3
        self.T_s3=T_s3

    def subframes_decode(self,symbols):
        #BCH decoding - Subframe1
        s1_received=symbols[:52]
        s1_decoded=self.bch_decode(s1_received)

        sub23=symbols[52:]
        #Deinterleaving Subframe2 and Subframe3
        k=46
        n=38
        deinterleave = lambda x,k,n: x.reshape(k,-1).T.flatten()
        Nav_deinter= deinterleave(sub23,k,n)

        #LDPC decoding - Subframe2
        s2_received=Nav_deinter[:1200]
        H=(self.H_transpose(self.A_s2,self.B_s2,self.C_s2,self.D_s2,self.E_s2,self.T_s2)).T
        s2_decoded=self.Ldpc_decode(H,s2_received)

        #LDPC decoding - Subframe3
        s3_received=Nav_deinter[1200:]
        H=(self.H_transpose(self.A_s3,self.B_s3,self.C_s3,self.D_s3,self.E_s3,self.T_s3)).T
        s3_decoded=self.Ldpc_decode(H,s3_received)

        return s1_decoded,s2_decoded[:600],s3_decoded[:274]

    def H_transpose(self,A,B,C,D,E,T):
        """ Generating Parity-Check Transpose matrix"""
        k=np.concatenate((A.T,B.T,T.T),axis=0) 
        s=np.concatenate((C.T,D.T,E.T),axis=0)
        H=np.concatenate((k,s),axis=1)
        return H
    
    def bch_decode(self,received_codeword):

        possible_codewords=[]
        for decimal_value in range(1, 401):
            binary_string = bin(decimal_value)[2:].zfill(9) 
            messages =  np.array([int(bit) for bit in binary_string])
            encoded_sample=self.instance.bch_encode(messages)
            possible_codewords.append(tuple(encoded_sample))
        
        # Calculate Hamming distance
        distances = [np.sum(np.array(received_codeword) != np.array(codeword)) for codeword in possible_codewords]

        # Find the index of the codeword with minimum distance
        min_distance_index = distances.index(min(distances))

        # Extract the decoded message part
        decoded_message = possible_codewords[min_distance_index]
        decoded = decoded_message[-9:]
        return decoded
    
    def Ldpc_decode(self,H,received_codeword):

        bpd=bp_decoder(H,error_rate=1/1200,bp_method="product_sum")
        decoded=bpd.decode(received_codeword)
        return decoded
    
    def bitstring(self,x):  return bin(x)[2:]
    def printlongdiv(self,lhs, rhs):
        rem = lhs
        div = rhs
        origlen = len(self.bitstring(div))
 
    # first shift left until the leftmost bits line up.
        count = 1
        while (div | rem) > 2*div:
            div <<= 1
            count += 1
 
    # now keep dividing until we are back where we started.
        quot = 0
        while count>0:
            quot <<= 1
            count -= 1
                #print("%14s" % bitstring(rem))
            divstr = self.bitstring(div)
            if (rem ^ div) < rem:
                quot |= 1
                rem ^= div
 
                    #print(1, " " * (11-len(divstr)), divstr[:origlen])
                #else:
                    #print(0, " " * (11-len(divstr)), "0" * origlen)
                #print(" " * (13-len(divstr)), "-" * origlen)
            div >>= 1
        return print("%14s <<< remainder" % self.bitstring(rem))
