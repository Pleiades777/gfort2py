# SPDX-License-Identifier: GPL-2.0+
import ctypes

try:
    import bigfloat as bf
    has_bf = True
except ImportError:
    print("Bigfloat not installed thus quad precision not fully supported")
    has_bf = False

# Precompute powers of two
if has_bf:
    pow2bf = []
    for i in range(123):
        pow2bf.append(bf.BigFloat(2,bf.quadruple_precision)**(-((i+1))))


def bytes2quad(bytearr):
    bb=[]
    for i in bytearray(bytearr)[::-1]:
        bb.append(bin(i)[2:].rjust(8,'0'))

    bb=''.join(bb)
    sign = bb[0]
    exp = bb[1:16]
    sig = bb[16:]

    a = int(exp,base=2)
    if a == 0:
        a = -16382
        b = bf.BigFloat(0,bf.quadruple_precision) 
    else:
        a = a-16383
        b = bf.BigFloat(1,bf.quadruple_precision) 

    for idx,i in enumerate(sig):
        b = b + bf.BigFloat(i)*pow2bf[idx]
    r = bf.BigFloat(2)**a * b
    if sign == '0':
        return r
    else:
        return -r

def quad2bytes(quad):
    if not isinstance(quad,bf.BigFloat):
        quad = bf.BigFloat(quad,bf.quadruple_precision)

    ba=[0]*16
    if not quad.hex() == bf.BigFloat(0).hex():

        bb = [0]*128
        if quad < 0:
            sign ='1'
        else:
            sign = '0'
        quad = bf.abs(quad)
        #print(quad)
        bb[0] = sign


        exp = int(bf.BigFloat(bf.log2(quad)))

        if exp <= -16382:
            #print(exp)
            exp = -16382
            sig = (quad /( 2**exp))
            #print(sig)
            bb[1:16] = '0'*15
        else:
            #print(exp)
            sig = bf.abs((quad /( 2**exp)) -1)
            #print(sig)
            bb[1:16] = bin(exp+16383)[2:].ljust(15,'0')

        for i in range(0,112):
            if sig >= pow2bf[i]:
                bb[i+16] = '1'
                sig = sig - pow2bf[i]
            else:
                bb[i+16] = '0'
            #print(x,bb[i+16],sig)

        bb=''.join(bb)
        for i in range(16):
            ba[i] = int(bb[i*8:(i+1)*8].ljust(8,'0'),base=2)

        ba=ba[::-1]


    c = ctypes.c_ubyte * 16
    cc = c()
    #print(bb)
    for i in range(16):
        cc[i] = ba[i]

    return cc

