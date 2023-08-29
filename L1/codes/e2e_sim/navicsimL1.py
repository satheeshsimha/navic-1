import numpy as np
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
# PRN Sequence generation API
#R0 regiser initial parameter for Data signal
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

class NavicL5sModulator():
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

        self.bitStream = np.empty((0, numChannel))
        self.datStream = np.empty((0, numChannel))
        full_frame = np.empty(0)
        full_data = np.empty(0)
        for i in range(numChannel):
            frame, data = self.__frameGen()
            full_frame = np.append(full_frame,frame)
            full_data = np.append(full_data,data)

        self.bitStream = np.append(self.bitStream,full_frame.reshape(numChannel,-1).T,axis=0)
        self.datStream = np.append(self.datStream,full_data.reshape(numChannel,-1).T,axis=0)


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
          genStream = np.append(genStream, np.repeat(self.bitStream[self.bitcnt: self.bitcnt+1,:], bufferCnt, axis=0), axis=0)
          # Update current bit's remaining duration
          self.samplesToNextBit -= bufferCnt
          # End loop
          bufferCnt = -1
        else:
          # Copy current bit for remaining duration
          genStream = np.append(genStream, np.repeat(self.bitStream[self.bitcnt: self.bitcnt+1,:], self.samplesToNextBit, axis=0), axis=0)
          # Increment bit counter
          self.bitcnt+=1
          # If current frame ended, generate new frame
          if(self.bitcnt%self.numSymbolsPerFrame==0):
            full_frame = np.empty(0)
            full_data = np.empty(0)
            for i in range(self.numChannel):
                frame, data = self.__frameGen()
                full_frame = np.append(full_frame,frame)
                full_data = np.append(full_data,data)

            self.bitStream = np.append(self.bitStream,full_frame.reshape(self.numChannel,-1).T,axis=0)
            self.datStream = np.append(self.datStream,full_data.reshape(self.numChannel,-1).T,axis=0)
          # Update remaining samples to generate
          bufferCnt -= self.samplesToNextBit
          # Update remaining duration of current bit
          self.samplesToNextBit = self.numSamplesPerBit
      
      return genStream[1:numBitsToGen+1]
    
    def __frameGen(self):
        """Function to add CRC, tail bits, interleave and encode the data bits

        :returns list frame: encoded symbols 
        :returns list nav_data: pre-encoded data

        """
        nav_data = np.array([])
        frame = np.array([],dtype=int)
        for i in range(3): 
            #data = np.array([np.random.randint(0, 2) for _ in range(276)])
            data = np.array([np.random.randint(0, 2) for _ in range(576)])
            cr = rtk_crc24q(data, len(data))
            crc = "{:06X}".format(cr)
            binary = bin(int(crc,16))[2:]
            padded_binary = list(binary.zfill(len(crc) * 4))
            nav_crc = np.append(data,padded_binary)
            nav_crc = np.array([int(bit) for bit in nav_crc])

        #fec encoding and tail bits
            #nav_crc_tail = np.append(nav_crc,np.zeros(6)) #tail bits
            #nav_data = np.append(nav_data,nav_crc_tail)
            nav_data = np.append(nav_data,nav_crc)
            #state = '000000'
            #cc1 = fec.FECConv(('1111001','1011011'))
            #nav_encd,state = cc1.conv_encoder(nav_crc,state)

        #interleaving
            #k=8
            #n= 75
            #interleave = lambda x,k,n: x.reshape(n,-1).T.flatten()
            #nav_intrlv = interleave(nav_encd,k,n)

        #adding sync word EB90 Hex
            #sync_hex = 'EB90'
            #sync_bin = bin(int(sync_hex,16))[2:]
            #sync_bin = [int(bit) for bit in sync_bin]
            #encsubframe = np.append(sync_bin,nav_intrlv)
            #frame = np.append(frame,encsubframe)
            #frame = np.append(frame,nav_intrlv)
            frame = np.append(frame, nav_crc)
        return frame, nav_data


    def GetBitStream(self):
       """Function to return bitstream of nav data
             
       :returns list genStream: generated bits

       """
       numBitsGen = (self.bitcnt//self.numSymbolsPerSubFrame)*self.numDataBitsPerSubFrame
       remain = self.bitcnt%self.numSymbolsPerSubFrame
       if(remain> 0 ):
          numBitsGen += (remain)//2
       return self.datStream[0:numBitsGen]



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

# x - input samples
# SqrtPr - square root of received power
def PowerScale(x, SqrtPr):
    rmsPow = np.sqrt(np.mean(np.abs(x)**2, axis=0))
    rmsPow[rmsPow==0.0] = 1
    scaledsig = SqrtPr*x/rmsPow
    return scaledsig

# Acquisition and Tracking API

def navic_pcps_acquisition(x, prnSeq_pilot, fs, fSearch, threshold=0):

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
    
    K = x.shape[0]
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

    Rxd = np.empty((K, N), dtype=np.complex_)
    for i in range(0,N):
        x_iq = x*np.exp(-1j*2*np.pi*fSearch[i]*t)
        XFFT = np.fft.fft(x_iq)
        YFFT = XFFT*prnSeq_pilot_FFT
        Rxd[:,i] = (1/K)*np.fft.ifft(YFFT)

    maxIndex = np.argmax(np.abs(Rxd)**2)
    maxCol = maxIndex%N
    maxRow = maxIndex//N
   
    powIn = np.mean(np.abs(x)**2)
    
    #plt.plot(np.abs(Rxd)**2)
    #plt.ylim([0,0.05])
    #plt.xlabel('time') ; plt.ylabel('Nav Data')
    #plt.show()
    
    
    sMax = np.abs(Rxd[maxRow, maxCol])**2
    thresholdEst = 2*K*sMax/powIn

    if(thresholdEst > threshold):
        tau = maxRow
        fDev = fSearch[maxCol]
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
              rho_cap =  (estimated_sig_power)/(estimated_total_power - estimated_sig_power)
              CN0_cap = 10* math.log10(rho_cap) + 10*math.log10(self.sample_rate/2.0) - 10*math.log10(self.code_length)
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
    def __init__(self, prnId):
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
        
        
        self.dataCodeLength = dataCodeLength
        #self.pilotCodeLength = pilotCodeLength
        #self.pilotOverlayCodeLength = pilotOverlayCodeLength
        #self.symbolRate = symbolRate
        self.codeTable_data = genNavicCaCode_Data(self.PRNID).astype(float)
        self.codeTable_pilot = genNavicCaCode_Pilot(self.PRNID).astype(float)
        #self.codeTable_pilot_overlay = genNavicCaCode_Pilot_Overlay(self.PRNID)
        
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

        
        
        self.pPromptCode_pilot = pilotSig*subCarrier # added subcarrier multipication
        self.pPromptCode_data = dataCodeSig * subCarrier

        ######################
        self.pSamplesPerChip = self.SampleRate / self.ChipRate
        sampleFactor = Fraction(self.pSamplesPerChip)
        #upSampleFactor = sampleFactor.numerator; downSampleFactor = sampleFactor.denominator
        self.numSamplesPerCodeBlock = self.SampleRate * 10e-3 # As each code block is of 10e-3 seconds
        
        
    
    def setupImpl(self):

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
        fc = self.CenterFrequency + self.InitialDopplerShift - self.pFLLNCOOut
        t = np.arange(self.pNumIntegSamples+1)/self.SampleRate
        phases = (2*np.pi*fc*t + self.pPreviousPhase - self.pPLLNCOOut) %(2*np.pi)
        #iqsig = u * np.exp(-1j*phases[:-1])
        iqsig = localbuf * np.exp(-1j*phases[1:])
        #iqsig = localbuf * np.exp(-1j*phases[:-1])
        
        self.pPreviousPhase = (phases[-1] + self.pPLLNCOOut)%(2*np.pi)
        
        numSamplesPerHalfChip = round(self.pSamplesPerChip*0.15)
        numSamplesPerThreeQuarterChip = round(self.pSamplesPerChip*0.6)
        
        # Code wipe-off Pilot
        # Update the prompt code appropriately
        iq_e_pilot = iqsig * np.roll(self.pPromptCode_pilot, -1*numSamplesPerHalfChip) # Early
        iq_ve_pilot = iqsig * np.roll(self.pPromptCode_pilot, -1*numSamplesPerThreeQuarterChip)# Very Early
        iq_p_pilot = iqsig * self.pPromptCode_pilot # Prompt for Pilot
        iq_l_pilot = iqsig * np.roll(self.pPromptCode_pilot, numSamplesPerHalfChip) # Late
        iq_vl_pilot = iqsig * np.roll(self.pPromptCode_pilot, numSamplesPerThreeQuarterChip) # Very Late
        
        integeval_pilot = np.mean(iq_e_pilot)
        integlval_pilot = np.mean(iq_l_pilot)
        integveval_pilot = np.mean(iq_ve_pilot)
        integvlval_pilot = np.mean(iq_vl_pilot)
        
        #Code wipe off for Data Channel
        iq_e_data = iqsig * np.roll(self.pPromptCode_data, -1*numSamplesPerHalfChip) # Early
        iq_ve_data = iqsig * np.roll(self.pPromptCode_data, -1*numSamplesPerThreeQuarterChip)# Very Early
        iq_p_data = iqsig * self.pPromptCode_data # Prompt for Data channel
        iq_l_data = iqsig * np.roll(self.pPromptCode_data, numSamplesPerHalfChip) # Late
        iq_vl_data = iqsig * np.roll(self.pPromptCode_data, numSamplesPerThreeQuarterChip) # Very Late
        
        integeval_data = np.mean(iq_e_data)
        integlval_data = np.mean(iq_l_data)
        integveval_data = np.mean(iq_ve_data)
        integvlval_data = np.mean(iq_vl_data)

        #Extract Data for Pilot Channel
        millisecdata_pilot = iq_p_pilot.reshape((int(self.PLLIntegrationTime/10), -1)).T # Each column contains ten millisecond of data 
        y_pilot = np.mean(millisecdata_pilot, axis=0) # Each element contains integrated value of one millisecond of data
        #y_pilot = np.mean(millisecdata_pilot, axis=0)
        integpval_pilot = np.sum(y_pilot)
        if len(iq_p_pilot) % 2 != 0: # Odd number of samples
            fllin_pilot = np.mean(np.reshape(np.concatenate([iq_p_pilot, [0]]), (2, -1)).T, axis=0) # Append a zero
        else:
            fllin_pilot = np.mean(iq_p_pilot.reshape((2, -1)).T, axis=0)

        #Extract Data from Data Channel
        millisecdata_data = iq_p_data.reshape((int(self.PLLIntegrationTime/10), -1)).T # Each column contains ten millisecond of data
        
        y_data = np.mean(millisecdata_data, axis=0) # Each element contains integrated value of one millisecond of data
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
        delaynco = self.pDLLNCOOut + integtime*loopfilterout
        self.pDLLNCOOut = delaynco
        
        fqyerr = 0
        fqyerr_pilot =0
        fqyerr_data = 0
        '''
        if(self.previous_prompt_pilot != None):
            
            
            phasor_pilot = np.conj(self.previous_prompt_pilot)*y_pilot
            fqyerr_pilot = -np.angle(phasor_pilot)/(2*np.pi*integtime)
            
        if(self.previous_prompt_data != None):
            
            
            phasor_data = np.conj(self.previous_prompt_data)*y_data
            fqyerr_data = -np.angle(phasor_data)/(2*np.pi*integtime)
        
        '''
        # FLL discriminator Pilot
        phasor_pilot = np.conj(fllin_pilot[0])*fllin_pilot[1]
        # phasor = np.conj(self.pPreviousIntegPVal)*integpval
          # angle is given between -pi and +pi
        fqyerr_pilot = -np.angle(phasor_pilot)/(np.pi*integtime)
        
        # FLL discriminator Data
        phasor_data = np.conj(fllin_data[0])*fllin_data[1]
        # phasor = np.conj(self.pPreviousIntegPVal)*integpval
        fqyerr_data = -np.angle(phasor_data)/(np.pi*integtime)
        
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
        self.pFLLNCOOut = fqynco
        
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
        self.pPLLNCOOut = phnco

        #self.updatePromptCode() ##Update the prompt code for next set of sample
        
        
        
        return y_pilot, y_data, fqyerr, fqynco, pherr, phnco, delayerr, delaynco, fc

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
        self.pBuffer_data = np.zeros(round(self.pNumSamplesToAppend))
        self.pFLLWPrevious1 = 0
        self.pFLLWPrevious2 = 0
        self.pFLLNCOOut = 0
        self.pPLLWPrevious1 = 0
        self.pPLLWPrevious2 = 0
        self.pPLLNCOOut = 0
        self.pDLLWPrevious1 = 0
        self.pDLLNCOOut = 0
#end of acquisition and tracking code

#function for bit synchronization index
def gnss_bit_synchronize(data, n):
    """Bit synchronization for IRNSS receiver
    
    :param list data: samples from tracking loop
    :param int n: samples per bit
    :returns int syncidx: returns the bit starting index
    
    """
    # Input validation
    numdata = len(data)
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError("Invalid input type for 'data'. Expected ndarray or list.")
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("Invalid input type or value for 'n'. Expected positive integer.")
    if numdata < n:
        raise ValueError("Invalid input length. 'data' should have length greater than or equal to 'n'.")

    # Find the transition locations in the data by finding the difference of signs in adjacent data
    all_signs = np.concatenate(([1], np.sign(data)))
    diff_vals = np.diff(all_signs)
    transition_flags = diff_vals != 0

    # Find the number of transitions at each data location
    ntemp = np.uint16(n)
    num_avg_samples = np.uint64(np.floor(numdata/ntemp) * ntemp)
    trmat = np.reshape(transition_flags[:num_avg_samples], (-1, ntemp)).T

    # Consolidate the number of transitions at each sample location
    numtr = np.sum(trmat, axis=1)

    # Find the starting location corresponding to maximum transitions and cast the value
    syncidx = np.argmax(numtr, axis=0)
    syncidx = np.uint16(syncidx)

    return syncidx

#function for frame synchronization index
SYNC_WORD = [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]  # Sync word value
INV_SYNC_WORD = [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1] # Inverted Sync word value (to check in case of inverted bits)
def find_sync_word(data):
    """Frame synchronization for IRNSS receiver
    
    :param list data: samples from tracking loop
    :returns int status: indicator for presence of sync word
    :returns int fsync_index: returns the frame starting index
    
    """
    sync_word_length = len(SYNC_WORD)
    window = []
    fsync_index = -1
    status = 0

    for index, bit in enumerate(data):
        window.append(bit)
        if len(window) > sync_word_length:
            window.pop(0)
        if window == SYNC_WORD:
            status = 1
            fsync_index = index - sync_word_length + 1
            break
        elif window == INV_SYNC_WORD:
            status = -1
            fsync_index = index - sync_word_length + 1
            break

    return status, fsync_index

def decoder(subframe, num_sf):
    """Function decodes the subframes
    :param list subframe: subframes from the extracted frame after frame synchronization
    :param int num_sf: number of suframes found
    :return list decd_total: returns decoded nav_data

    """
    decd_total = np.empty(0)
    #cc1 = fec.FECConv(('1111001','1011011'))
    #remove sync word
    for buff in subframe:
        rec_subframe = buff[16:]

        k = 8
        n = 75

    #undo interleaving
        #deinterleave = lambda x,k,n: x.reshape(k,-1).T.flatten()
        #nav_deintrv = deinterleave(rec_subframe,k,n)
        #nav_deintrv = deinterleave(buff,k,n)
        #nav_deintrv = [int(bit) for bit in nav_deintrv]
        nav_deintrv = [int(bit) for bit in buff]
        nav_deintrv = np.asarray(nav_deintrv)
        #nav_deintrv = np.append(nav_deintrv,np.zeros(18)) #to nullify error in viterbi decoder


    #fec decoding and ber
        #yn_hard = ((np.sign(nav_deintrv.real)+1)/2).astype(int)
        #nav_decd = cc1.viterbi_decoder(yn_hard,'hard')
        #decd_total = np.append(decd_total,nav_decd) 
        decd_total = np.append(decd_total,nav_deintrv)
    return decd_total

#main program
#code chip rate, sample rate and sample period
#refer to navicsim.py for all function detials
dataCodeLength = 10230
pilotCodeLength = 10230
pilotOverlayCodeLength = 1800
codeFreqBasis = 1.023e6


sampleRate = 10*codeFreqBasis
samplePeriod = 1/sampleRate
symbolRate = 100
#satId is the satellite ID for multiple satellites to track
satId = np.array([25, 37, 63, 41])
#satId = np.array([25,41])
#satId = np.array([27])
numChannel = len(satId)


#frequrency shift to be applied to the signal
#fShift = np.array([489, 1299, 3796, 4888])
fShift = np.array([4853,4988,3868,1835])
channelpfo = PhaseFrequencyOffset(sampleRate)
#sigDelay is the delay in samples in channels
sigDelay = np.array([300.34, 587.21, 425.89, 312.88])
#sigDelay = np.array([425.89, 312.88])
dynamicDelayRange = 50
staticDelay = np.round(sigDelay - dynamicDelayRange)
channelstatd = IntegerDelay(staticDelay)
channelvard = FractionalDelay(1, 65535)

PLLIntegrationTime = 10e-3
PLLNoiseBandwidth = 18 # In Hz
#PLLNoiseBandwidth = 18 # In Hz
#FLLNoiseBandwidth = 4 # In Hz
FLLNoiseBandwidth = 2 # In Hz
DLLNoiseBandwidth = 1  # In Hz


#simulation duration, steps at which values are recorded(here for every 10ms)
simDuration = 6

#timeStep = 1
timeStep = PLLIntegrationTime
numSteps = round(simDuration/timeStep)
samplePerStep = int(timeStep/samplePeriod)


codeTable_data = genNavicCaTable_Data(sampleRate, dataCodeLength,codeFreqBasis, satId)
codeTableSampCnt_data = len(codeTable_data)


codeTable_pilot = genNavicCaTable_Pilot(sampleRate, pilotCodeLength,codeFreqBasis,  satId)
codeTableSampCnt_pilot = len(codeTable_pilot)


#codeTable_pilot_overlay = genNavicCaTable_Pilot_Overlay(sampleRate, pilotOverlayCodeLength,symbolRate,satId)
#codeTableSampCnt_pilot_overlay = len(codeTable_pilot_overlay)

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




sqrtPr = np.sqrt(Pt*DtLin*DrLin)*(1/(4*np.pi*(fe+fShift)*sigDelay*samplePeriod))



rms = lambda x: np.sqrt(np.mean(np.abs(x)**2, axis=0)) 

    
datagen = NavicDataGen(symbolRate, sampleRate, numChannel)
pilotOverlayCodegen = PilotOverlayBitGen(satId, pilotOverlayCodeLength, symbolRate, sampleRate)
modulator = NavicL5sModulator(sampleRate)
#istep = 0 
#np.set_printoptions(threshold=np.inf)
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
    #scaledSig = PowerScale(iqsig, sqrtPr)
    scaledSig = PowerScale(delayedSig, sqrtPr)
    
    # Add signals from each channel
    resultsig = np.sum(scaledSig, axis=1)
    #resultsig = np.sum(iqsig, axis=1)
    # Generate noise
    noisesig = (np.random.normal(scale=Nr**0.5, size=(samplePerStep, )) + 1j*np.random.normal(scale=Nr**0.5, size=(samplePerStep, )))/2**0.5

    # Add thermal noise to composite signal
    rxwaveform = resultsig + noisesig
    #rxwaveform = resultsig
    
    # Scale received signal to have unit power
    waveform = rxwaveform/rms(rxwaveform)  
    
    #np.set_printoptions(threshold=np.inf)

    #print("length=", len(waveform))

    # Perform acquisition once from cold-start
    if istep == 0:

        # Acqusition doppler search space
            fMin = -5000
            fMax = 5000
            fStep = 250
            fSearch = np.arange(fMin, fMax + fStep , fStep)

            tracker = []
            satVis = 0
        
        # Perform acquisition for each satellite
            for prnId in range(numChannel):
                status, codePhase, doppler = navic_pcps_acquisition(
                                            waveform, 
                                            codeTable_pilot[np.arange(0, samplePerStep)%codeTableSampCnt_pilot, prnId],
                                            sampleRate ,fSearch
                                        )   
                delaySamp = codePhase
                codePhase = (codePhase % codeTableSampCnt_data)/(sampleRate/codeFreqBasis)
            
                print(f"Acquisition results for PRN ID {satId[prnId]}\n Status:{status} Doppler:{doppler} Delay/Code-Phase:{delaySamp}/{codePhase}")

# If a satellite is visible, initialize tracking loop
                if(status == True):
                    satVis += 1 
                    tracker.append(NavicTracker(satId[prnId]))
                    tracker[-1].SampleRate = sampleRate
                    tracker[-1].CenterFrequency = 0
                    tracker[-1].PLLNoiseBandwidth = PLLNoiseBandwidth
                    tracker[-1].FLLNoiseBandwidth = FLLNoiseBandwidth
                    tracker[-1].DLLNoiseBandwidth = DLLNoiseBandwidth
                    tracker[-1].PLLIntegrationTime = round(PLLIntegrationTime*1e3)
                    tracker[-1].PRNID = satId[prnId]
                    tracker[-1].InitialDopplerShift = doppler
                    tracker[-1].InitialCodePhaseOffset = codePhase
                    tracker[-1].setupImpl()
                    tracker[-1].resetImpl()
            #trackDataShape = (numSteps*round(PLLIntegrationTime*1e3), satVis)
                trackDataShape = (round(numSteps), satVis)
                y_pilot = np.empty(trackDataShape, dtype=np.complex_)
                y_data = np.empty(trackDataShape, dtype=np.complex_)
                fqyerr = np.empty(trackDataShape)
                fqynco = np.empty(trackDataShape)
                pherr = np.empty(trackDataShape)
                phnco = np.empty(trackDataShape)
                delayerr = np.empty(trackDataShape)
                delaynco = np.empty(trackDataShape)
                fc = np.empty(trackDataShape)
           # input_phase = np.empty(trackDataShape)

    # Perform tracking for visible satellites
    for i in range(satVis):
            y_pilot[istep, i], y_data[istep,i],fqyerr[istep, i], fqynco[istep, i], pherr[istep, i], phnco[istep, i], delayerr[istep, i], delaynco[istep, i] ,fc[istep,i]= tracker[i].stepImpl(waveform)

#print(len(y_pilot))

k = len(y_data)    
np.set_printoptions(threshold=np.inf)

#print("y-pilot=", y_pilot)
#print("y_data=", y_data)
"""## Bit and Frame Synchronization"""
for i in range(satVis):
    #pilot_overlay_sent = pilotOverlayCodegen.GetBitStream()[:-1, i]
    #k = np.real(y_pilot[0:,i])
    n = 1 #Number of data per bit
    skip = 0 #Forgo few bits as the tracking loops starts early
    #k = np.real(y_data[n*skip:,i])
    h = np.real(y_data[n*skip:,i])
    #sync_index, num_tr = gnss_bit_synchronize(k, n)
    #sync_index = 0
    #print("Synchronization index:", sync_index)

    #l = np.mean(k[sync_index:(((len(k)-sync_index)//n) * n) + sync_index].reshape(-1,n).T, axis=0)
    mapbits = lambda l: np.piecewise(l, [l < 0, l >= 0], [1, 0])
    mapbits_inverted = lambda l: np.piecewise(l, [l < 0, l >= 0], [0, 1])
    bits = mapbits(h)
    bits_inverted = mapbits_inverted(h)
    #print(pilot_overlay_sent)
    #status, fsync_index = find_sync_word(bits)

    #print("Frame Sync status:", status)
    #print("Frame Sync:", fsync_index)
    #if status == -1:
      #  bits = 1*np.logical_not(bits)

    #sync_frames = bits[fsync_index:]
   
    
    print("satellite=",i)
    num_sf = len(bits)//600
    #navbits = datagen.GetBitStream()[0:600,0].reshape(-1,600)
    navbits = datagen.GetBitStream()
    #print("Navbits", navbits)
    #print("bits=", bits)
    #print("bit inverted=", bits_inverted)
    #print("pherr=", pherr[:, i])
    #if(np.array_equal(bits[1:], navbits[:-1])):
     #   print("Equal number of bits=", np.sum(np.equal(bits[1:], navbits[:-1])))
      #  print("Equal")
    #elif (np.array_equal(bits_inverted[1:], navbits[:-1])):
     #       print("Equal number of inverted bits=", np.sum(np.equal(bits_inverted[1:], navbits[:-1])))
            #print("bits inverted =", bits[k:(j+1)*600])
      #      print("Equal")
    #else :
     #       print("Not Equal")
    #
    #check = decoder(bits[0:num_sf*600].reshape(-1,600),num_sf).reshape(-1,600)
    for iter in range(k) :
            print("iter=", iter, navbits[iter,i]==bits[iter], navbits[iter,i]==bits_inverted[iter], fc[iter,i],fqyerr[iter,i],fqynco[iter,i] )
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

'''
    for j in range(num_sf):print("pherr=", pherr[k:(j+1)*600:, i])
        print("j=", j)
        #print("navbits=",navbits[j,:])
        k = j*600
        #
        if(np.array_equal(bits[k:(j+1)*600], navbits[j,:])):
            print("Equal number of bits=", np.sum(np.equal(bits[k:(j+1)*600], navbits[j,:])))
            print("Equal")
            #print("bits=", bits[k:(j+1)*600])
        elif (np.array_equal(bits_inverted[k:(j+1)*600], navbits[j,:])):
            print("Equal number of inverted bits=", np.sum(np.equal(bits_inverted[k:(j+1)*600], navbits[j,:])))
            #print("bits inverted =", bits[k:(j+1)*600])
            print("Equal")
        else :
            print("Not Equal")



mapbit = lambda y: np.piecewise(np.real(y), [np.real(y) < 0, np.real(y) >= 0], [1, 0])
mapbitinv = lambda y: np.piecewise(np.real(y), [np.real(y) < 0, np.real(y) >= 0], [0, 1])

map_pilot_overlay = lambda y: np.piecewise(np.imag(y), [np.imag(y) < 0, np.imag(y) >= 0], [1, 0])
map_pilot_overlay_inv = lambda y: np.piecewise(np.imag(y), [np.imag(y) < 0, np.imag(y) >= 0], [0, 1])
 
#np.set_printoptions(threshold=np.inf)
    
#print(mapbit)

pilot_overlay_sent = pilotOverlayCodegen.GetBitStream()[:-1, 0]
y_regular = map_pilot_overlay(y_pilot[:, 0])
y_inverted = map_pilot_overlay_inv(y_pilot[:, 0])

y_data_regular = mapbit(y_data[:, 0])
y_data_inverted = mapbitinv(y_data[:, 0])

data_sent = datagen.GetBitStream()[:-1, 0]

print("Pilot true=",np.sum(np.equal(y_regular, pilot_overlay_sent)))

for iter in range(k-1): 
    #print("data sent =", data_sent[iter], "data reg=", y_data_regular[iter], "data inverted=", y_data_inverted[iter])
    print(iter,"pilot sent=", pilot_overlay_sent[iter],"pilot sec code reg=", y_regular[iter+1], "inverted=", y_inverted[iter+1], pilot_overlay_sent[iter] == y_regular[iter+1],y_pilot[iter+1,0])
    #print(iter,"data sent=", data_sent[iter],"data reg=", y_data_regular[iter], "inverted=", y_data_inverted[iter], data_sent[iter] == y_data_regular[iter],y_data[iter,0])

for iter in range(k-1): 
    #print("data sent =", data_sent[iter], "data reg=", y_data_regular[iter], "data inverted=", y_data_inverted[iter])
    #print(iter,"pilot sent=", pilot_overlay_sent[iter],"pilot sec code reg=", y_regular[iter+1], "inverted=", y_inverted[iter+1], pilot_overlay_sent[iter] == y_regular[iter+1],y_pilot[iter+1,0])
    print(iter,"data sent=", data_sent[iter],"data reg=", y_data_regular[iter+1], "inverted=", y_data_inverted[iter+1], data_sent[iter] == y_data_regular[iter+1],y_data[iter+1,0])

#print("BER wrt regular = ", np.sum(np.equal(data_sent[0:k],y_data_regular))/len(data_sent)*100)
#print("BER wrt inverted = ", np.sum(np.equal(data_sent[0:k], y_data_inverted) )/len(data_sent)*100)

#print("Received y bits:\n", mapbit(y[:, 0])[::-1])
#print("Received y bits inverted:\n", mapbitinv(y[:, 0])[::-1])

#print("Received y-data bits:\n", mapbit(y_data[:, 0])[::-1])
#print("Received y- data bits inverted:\n", mapbitinv(y_data[:, 0])[::-1])

#print("Transmitted Bits:\n",datagen.GetBitStream()[::-1, 0])


'''

