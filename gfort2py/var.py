# SPDX-License-Identifier: GPL-2.0+
import builtins
import ctypes
import numpy as np

try:
    import bigfloat as bf
    has_bf = True
except ImportError:
    print("Bigfloat not installed quad precision not fully supported")
    has_bf = False

from .errors import IgnoreReturnError


class fVar():
    def __init__(self, obj):
        self.var = obj['var']
        if 'mangled_name' in obj:
            self.mangled_name = obj['mangled_name']
        if 'name' in obj:
            self.name = obj['name']

        self.ctype = self.var['ctype']
        self.pytype = self.var['pytype']

        if self.pytype == 'quad':
            self.ctype = ctypes.c_byte * 16
        elif self.pytype == 'bool':
            self.pytype = bool
            self.ctype = ctypes.c_int32
        else:
            self.pytype = getattr(builtins, self.pytype)
            self.ctype = getattr(ctypes, self.ctype)


    def convert_to_py(self,ctype):
        if self.pytype == 'quad':
            if has_bf:
                return _bytes2quad(ctype)
            else:
                raise ValueError("Must install BigFloat first")
        else:
            if hasattr(ctype, 'value'):
                return self.pytype(ctype.value)
            else:
                return self.pytype(ctype)


    def convert_from_py(self,pytype):
        if self.pytype == 'quad':
            if has_bf:
                return _quad2bytes(pytype)
            else:
                raise ValueError("Must install BigFloat first")
        else:
            return self.ctype(self.pytype(pytype))

    def from_address(self, addr):
        """
        Given an address, return a ctype object from that address.

        addr -- integer address

        Returns:
        A ctype representation of the variable
        """
        return self.convert_to_py(self.ctype.from_address(addr).value)

    def set_from_address(self, addr, value):
        """
        Given an address set the variable to value

        addr -- integer address
        value -- python object that can be coerced into self.pytype

        """
        self.ctype.from_address(addr).value = self.convert_from_py(value).value

    def from_param(self, value):
        """
        Returns a ctype object needed by a function call

        This is a pointer to self.ctype(value).

        If the function argument is optional, and value==None we return None
        else we return the pointer.

        value -- python object that can be coerced into self.pytype

        """

        if 'optional' in self.var:
            if self.var['optional'] and value is None:
                return None

        ct = self.convert_from_py(value)

        if 'value' in self.var and self.var['value']:
            return ct
        else:
            ct = ctypes.pointer(ct)
            if 'pointer' in self.var and self.var['pointer']:
                return ctypes.pointer(ct)
            else:
                return ct

    def from_func(self, pointer):
        """
        Given the pointer to the variable return the python type

        We raise IgnoreReturnError when we get something that should not be returned,
        like the hidden string length that's at the end of the argument list.

        """
        x = pointer
        while hasattr(x, 'contents'):
            x = x.contents

        if hasattr(x, 'value'):
            x = x.value

        if x is None:
            return None

        try:
            return self.convert_to_py(x)
        except AttributeError:
            raise IgnoreReturnError

    def in_dll(self, lib):
        return self.convert_to_py(self.ctype.in_dll(lib, self.mangled_name))

    def set_in_dll(self, lib, value):
        c = self.ctype.in_dll(lib, self.mangled_name)
        if hasattr(c, 'value'):
            self.ctype.in_dll(lib, self.mangled_name).value = self.convert_from_py(value).value
        else:
            cc = self.convert_from_py(value)
            for i in range(len(c)):
                c[i] = cc[i]


class fParam():
    def __init__(self, obj):
        self.param = obj['param']
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype == 'bool':
            self.pytype = bool
            self.ctype = 'c_int32'
        else:
            self.pytype = getattr(builtins, self.pytype)

    def set_in_dll(self, lib, value):
        """
        Can't set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def in_dll(self, lib):
        """
        A parameters value is stored in the objects dict, as we can't access them
        from the shared lib.
        """
        return self.pytype(self.value)



# Precompute powers of two
if has_bf:
    pow2bf = []
    for i in range(123):
        pow2bf.append(bf.BigFloat(2,bf.quadruple_precision)**(-((i+1))))


def _bytes2quad(bytearr):
    bb=[]
    for i in bytearray(bytearr)[::-1]:
        bb.append(bin(i)[2:].rjust(8,'0'))

    bb=''.join(bb)

    sign = bb[0]
    exp = bb[1:16]
    sig = bb[16:]

    a = int(exp,base=2)-16383
    b = bf.BigFloat(1,bf.quadruple_precision) 
    for idx,i in enumerate(sig):
        b = b + int(i)*pow2bf[idx]
    r = 2**a * b
    if sign == '0':
        return r
    else:
        return -r


def _quad2bytes(quad):
    if not isinstance(quad,bf.BigFloat):
        quad = bf.BigFloat(quad,bf.quadruple_precision)

    ba=[0]*16
    if not quad==bf.BigFloat(0):
        bb = [0]*128
        if quad < 0:
            sign ='1'
        else:
            sign = '0'
        #print(quad)
        bb[0] = sign

        bb = [0]*128
        if quad < 0:
            sign ='1'
        else:
            sign = '0'
        #print(quad)
        bb[0] = sign

        exp = int(bf.BigFloat(bf.log2(quad)))
        #print(exp)
        sig = (quad /( 2**exp)) -1
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

    c = ctypes.c_byte * 16
    cc = c()
    #print(bb)
    for i in range(16):
        cc[i] = ba[i]

    return cc


