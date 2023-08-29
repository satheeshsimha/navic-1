import numpy as np
from fractions import Fraction

import scipy.constants as sciconst
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.signal

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
    a = np.array(cad)
    return a
    #return  np.array(cad)   
 
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
def genNavicCaTable_Data(samplingFreq, prnIdMax):
   # prnIdMax = 64
    codeLength = 10230
    codeFreqBasis = 1.023e6
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    #print(sampleCount)
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float32)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    #print(indexArr)
    return np.array([genNavicCaCode_Data(i) for i in range(1,prnIdMax+1)])[:,indexArr].T

#function to upsample the Pilot PRN sequence generated to required sampling rate
def genNavicCaTable_Pilot(samplingFreq, prnIdMax):
    #prnIdMax = 64
    codeLength = 10230
    codeFreqBasis = 1.023e6
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float32)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    return np.array([genNavicCaCode_Pilot(i) for i in range(1,prnIdMax+1)])[:,indexArr].T

#function to upsample the Pilot Overlay PRN sequence generated to required sampling rate
def genNavicCaTable_Pilot_Overlay(samplingFreq, prnIdMax):
  #  prnIdMax = 64
    codeLength = 1800
    codeFreqBasis = 100
    samplingPeriod = 1/samplingFreq
    sampleCount = int(np.round(samplingFreq / (codeFreqBasis / codeLength)))
    indexArr = (np.arange(sampleCount)*samplingPeriod*codeFreqBasis).astype(np.float64)     # Avoid floating point error due to high precision
    indexArr = indexArr.astype(int)
    return np.array([genNavicCaCode_Pilot_Overlay(i) for i in range(1,prnIdMax+1)])[:,indexArr].T

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
        codeNumSample_pilot_overlay = codeTable3.shape[0]
        
        numSample = x.shape[0]
        numChannel = x.shape[1]
        
        codeTable_p = codeTable2[(self.prnstart+np.arange(numSample))%codeNumSample_pilot]
        codeTable_o = codeTable3[(self.cstart+np.arange(numSample))%codeNumSample_pilot_overlay]
        
        self.prnstart = ( self.prnstart+numSample)%codeNumSample_pilot
        self.cstart = ( self.cstart+numSample)%codeNumSample_pilot_overlay
        
        # Subcarrier generation for BOC
        subCarr1Ch1 = self.__GenBocSubCarrier(numSample, self.m1)
        subCarr1Ch2 = self.__GenBocSubCarrier(numSample, self.m2)
        
        SubCarrSig1 = np.tile(np.array([subCarr1Ch1]).T, (1, numChannel))
        SubCarrSig2 = np.tile(np.array([subCarr1Ch2]).T, (1, numChannel))
        
        
        # PRN sequence of SPS is RS pilot PRN sequence
        PilotCode = (codeTable_p+codeTable_o)%2
        PilotSig = 1-2*PilotCode
        # Data for RS is data of SPS
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
        

        return iqsig
    
    def __GenBocSubCarrier(self, N, m):
        ts = 1/self.sampleRate
        t = np.arange(N)*ts
        fsc = m*1.023e6
         
        if(m == 1):
            subCarrier = np.sign(np.sin(2*np.pi*(fsc*t + self.subCarrPhase1)))
            self.subCarrPhase1 += fsc*N*ts
            self.subCarrPhase1 -= int(self.subCarrPhase1)
        if(m == 6):
            subCarrier = np.sign(np.sin(2*np.pi*(fsc*t + self.subCarrPhase2)))
            self.subCarrPhase2 += fsc*N*ts
            self.subCarrPhase2 -= int(self.subCarrPhase2)
        return subCarrier
    
    def Release(self,m):
        self.codePhase = 0

        fsc = m*1.023e6
        epsilon = fsc*1/(100*self.sampleRate)
        self.subCarrPhase = epsilon

#class to generate navigation data at 100sps
class NavicDataGen():
    def __init__(self, ds=100, fs=10*1.023e6, numChannel=1, file=None):
      self.dataRate = ds
      self.sampleRate = fs
      self.numSamplesPerBit = round(fs/ds)
      self.samplesToNextBit = self.numSamplesPerBit
      self.numChannel = numChannel
      self.bitStream = np.empty((1, numChannel))
      self.bitStream[0,:] = np.random.binomial(1, 0.5, (numChannel, ))

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
          self.bitStream = np.append(self.bitStream, np.random.binomial(1, 0.5, (1, self.numChannel)), axis=0)
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

def navic_pcps_acquisition(x, prnSeq_data, fs, fSearch, threshold=0):

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
    fsc = 1.023e6  # Subcarrier1 - 1.023MHz
    subCarrier = np.sign(np.sin(2*np.pi*(fsc*t)))
    prnSeq_data_FFT = np.conjugate(np.fft.fft(((1-2*prnSeq_data)*subCarrier)))
    
   
    N = fSearch.shape[0]
   # ts = 1/fs
    #t = np.arange(K)*ts

    Rxd = np.empty((K, N), dtype=np.complex_)
    for i in range(0,N):
        x_iq = x*np.exp(-1j*2*np.pi*fSearch[i]*t)
        XFFT = np.fft.fft(x_iq)
        YFFT = XFFT*prnSeq_data_FFT
        Rxd[:,i] = (1/K)*np.fft.ifft(YFFT)

    maxIndex = np.argmax(np.abs(Rxd)**2)
    maxCol = maxIndex%N
    maxRow = maxIndex//N

    powIn = np.mean(np.abs(x)**2)
    
    plt.plot(np.abs(Rxd)**2)
    plt.ylim([0,0.05])
    #plt.xlabel('time') ; plt.ylabel('Nav Data')
    plt.show()
    
    
    sMax = np.abs(Rxd[maxRow, maxCol])**2
    thresholdEst = 2*K*sMax/powIn

    if(thresholdEst > threshold):
        tau = maxRow
        fDev = fSearch[maxCol]
        return True, tau, fDev
    else:
        return False, 0, 0      




#main program
#code chip rate, sample rate and sample period
#refer to navicsim.py for all function detials
codeFreqBasis = 1.023e6
sampleRate = 6*codeFreqBasis
samplePeriod = 1/sampleRate
symbolRate = 100
#satId is the satellite ID for multiple satellites to track
satId = np.array([1])
numChannel = len(satId)

#simulation duration, steps at which values are recorded(here for every 10ms)
simDuration = 10

#timeStep = 1
timeStep = 10e-3
numSteps = round(simDuration/timeStep)
samplePerStep = int(timeStep/samplePeriod)


codeTable_data = genNavicCaTable_Data(sampleRate,numChannel)
codeTableSampCnt_data = len(codeTable_data)


codeTable_pilot = genNavicCaTable_Pilot(sampleRate,numChannel)
codeTableSampCnt_pilot = len(codeTable_pilot)


codeTable_pilot_overlay = genNavicCaTable_Pilot_Overlay(sampleRate,numChannel)
codeTableSampCnt_pilot_overlay = len(codeTable_pilot_overlay)

c = sciconst.speed_of_light
fe = 1176.45e6;              
Dt = 12;                     
DtLin = 10*np.log10(Dt)
Dr = 4;                      
DrLin = 10*np.log10(Dr)
Pt = 44.8;                   
k = sciconst.Boltzmann;  
T = 300;                     
rxBW = 24e6;                 
Nr = k*T*rxBW;              


#frequrency shift to be applied to the signal
fShift = np.array([2568])
channelpfo = PhaseFrequencyOffset(sampleRate)
#sigDelay is the delay in samples in channels
sigDelay = np.array([312.88])
dynamicDelayRange = 50
staticDelay = np.round(sigDelay - dynamicDelayRange)
channelstatd = IntegerDelay(staticDelay)
channelvard = FractionalDelay(1, 65535)

sqrtPr = np.sqrt(Pt*DtLin*DrLin)*(1/(4*np.pi*(fe+fShift)*sigDelay*samplePeriod))

datagen = NavicDataGen(symbolRate, sampleRate, numChannel)
modulator = NavicL5sModulator(sampleRate)

rms = lambda x: np.sqrt(np.mean(np.abs(x)**2, axis=0)) 

istep = 0

#Navigation Data
navdata = datagen.GenerateBits(timeStep)


#Baseband modulation
iqsig = modulator.Modulate(navdata,codeTable_data[:,satId-1],codeTable_pilot[:,satId-1],codeTable_pilot_overlay[:,satId-1])

# Doppler shift
doppsig = channelpfo.Offset(iqsig, fShift)

# Delay
staticDelayedSignal = channelstatd.Delay(doppsig)
leftoutDelay = sigDelay - staticDelay
delayedSig = channelvard.Delay(staticDelayedSignal, leftoutDelay)

# Power scaling
scaledSig = PowerScale(delayedSig, sqrtPr)

# Add signals from each channel
resultsig = np.sum(scaledSig, axis=1)

# Generate noise
noisesig = (np.random.normal(scale=Nr**0.5, size=(samplePerStep, )) + 1j*np.random.normal(scale=Nr**0.5, size=(samplePerStep, )))/2**0.5

# Add thermal noise to composite signal
rxwaveform = resultsig + noisesig

# Scale received signal to have unit power
waveform = rxwaveform/rms(rxwaveform)  




#np.set_printoptions(threshold=np.inf)

print("length=", len(waveform))



# Perform acquisition once from cold-start
if istep == 0:

        # Acqusition doppler search space
        fMin = -5000
        fMax = 5000
        fStep = 500
        fSearch = np.arange(fMin, fMax, fStep)

        tracker = []
        satVis = 0
        
        # Perform acquisition for each satellite
        for prnId in satId:
            status, codePhase, doppler = navic_pcps_acquisition(
                                            waveform, 
                                            codeTable_data[np.arange(0, samplePerStep)%codeTableSampCnt_data, prnId-1],
                                            sampleRate ,fSearch
                                        )   
            delaySamp = codePhase
            codePhase = (codePhase % codeTableSampCnt_data)/(sampleRate/codeFreqBasis)

            print(f"Acquisition results for PRN ID {prnId}\n Status:{status} Doppler:{doppler} Delay/Code-Phase:{delaySamp}/{codePhase}")
