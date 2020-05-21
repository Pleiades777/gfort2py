# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
import sys

def make_complex(ct):

    class _complex(ctypes.Structure):
        _fields_ = [('real', ct),
                    ('imag', ct)]

    return _complex

################33



# gfortran 8 needs https://gcc.gnu.org/wiki/ArrayDescriptorUpdate
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin

# From gcc source code
# Parsed       Lower   Upper  Returned
# ------------------------------------
#:           NULL    NULL   AS_DEFERRED (*)
# x            1       x     AS_EXPLICIT
# x:           x      NULL   AS_ASSUMED_SHAPE
# x:y          x       y     AS_EXPLICIT
# x:*          x      NULL   AS_ASSUMED_SIZE
# *            1      NULL   AS_ASSUMED_SIZE

_index_t = ctypes.c_int64
_size_t = ctypes.c_int64
_mod_version = 15
_GFC_MAX_DIMENSIONS = 7
_GFC_DTYPE_RANK_MASK = 0x07
_GFC_DTYPE_TYPE_SHIFT = 3
_GFC_DTYPE_TYPE_MASK = 0x38
_GFC_DTYPE_SIZE_SHIFT = 6

_BT_UNKNOWN = 0
_BT_INTEGER = _BT_UNKNOWN + 1
_BT_LOGICAL = _BT_INTEGER + 1
_BT_REAL = _BT_LOGICAL + 1
_BT_COMPLEX = _BT_REAL + 1
_BT_DERIVED = _BT_COMPLEX + 1
_BT_CHARACTER = _BT_DERIVED + 1
_BT_CLASS = _BT_CHARACTER + 1
_BT_PROCEDURE = _BT_CLASS + 1
_BT_HOLLERITH = _BT_PROCEDURE + 1
_BT_VOID = _BT_HOLLERITH + 1
_BT_ASSUMED = _BT_VOID + 1

_BT_TYPESPEC = {_BT_UNKNOWN: 'v', _BT_INTEGER: 'i', _BT_LOGICAL: 'b',
            _BT_REAL: 'f', _BT_COMPLEX: 'c', _BT_DERIVED: 'v',
            _BT_CHARACTER: 'v', _BT_CLASS: 'v', _BT_PROCEDURE: 'v',
            _BT_HOLLERITH: 'v', _BT_VOID: 'v', _BT_ASSUMED: 'v'}

_PY_TO_BT = {'int': _BT_INTEGER, 'float': _BT_REAL, 'bool': _BT_LOGICAL,
        'str': _BT_CHARACTER, 'bytes': _BT_CHARACTER}

if sys.byteorder == 'little':
    _byte_order = ">"
else:
    _byte_order = "<"

class _bounds14(ctypes.Structure):
    _fields_ = [("stride", _index_t),
                ("lbound", _index_t),
                ("ubound", _index_t)]

class _dtype_type(ctypes.Structure):
    _fields_ = [("elem_len", _size_t),
                ('version', ctypes.c_int),
                ('rank', ctypes.c_byte),
                ('type', ctypes.c_byte),
                ('attribute', ctypes.c_ushort)]


def _make_fAlloc15(ndims):
    class _fAllocArray(ctypes.Structure):
        _fields_ = [('base_addr', ctypes.c_void_p),
                    ('offset', _size_t),
                    ('dtype', _dtype_type),
                    ('span', _index_t),
                    ('dims', _bounds14 * ndims)
                    ]
    return _fAllocArray






class arrayDescriptor(): 
    def __init__(self, ndims, elem):
        self.ndims = ndims
        self.ctype = _make_fAlloc15(self.ndims)
        self._ictype = None # Instance of self.ctype()
        self.elem = elem # The basic type (int, derived type, etc) of one unit of the array
        self._p = None

    def allocate(self):
        self._ictype = self.ctype()

    def isAllocated(self):
        if self._ictype is None:
            return False
        return self._ictype.base_addr is not None

    def pointer(self):
        if not self.isAllocated():
            self.allocate()
        self._p = ctypes.pointer(self._ictype)
        return self._p

    def in_dll(self, lib, name):
        self._ictype = self.ctype.in_dll(lib, name)
        return self

    def from_address(self, addr):
        self._ictype = self.ctype.from_address(addr)
        return self  

    def addressof(self):
        return ctypes.addressof(self._ictype)

    def from_param(self):
        if not self.isAllocated():
            self.allocate()
        return self._ictype

    def __getitem__(self, key):
        if not self.isAllocated():
            return

        if isinstance(key, slice):
            raise NotImplementedError()

        ind = self._index(key)

        addr = ctypes.addressof(self._ictype) + ind * ctypes.sizeof(self.elem)

        return self.elem.from_address(addr)


    def __setitem__(self, key, value):
        if not self.isAllocated():
            return

        if isinstance(key, slice):
            raise NotImplementedError()

        ind = self._index(key)

        addr = ctypes.addressof(self._ictype) + ind * ctypes.sizeof(self.elem)

        x = self.elem.from_address(addr)
        x.value = value

    def _index(self, key):
        if isinstance(key, tuple):
            if len(key) != self.ndims:
                raise IndexError("Wrong number of dimensions")
            ind = np.ravel_multi_index(key, self.shape)
        else:
            ind = key

        if ind > self.size:
            raise ValueError("Out of bounds")

        return ind

    @property
    def shape(self):
        if not self.isAllocated():
            return

        shape = []
        for i in range(self.ndims):
            shape.append(self._ictype.dims[i].ubound - self._ictype.dims[i].lbound + 1)

        return tuple(shape)     
        
    @shape.setter
    def shape(self, shape):
        for i in range(self.ndims):
            self._ictype.dims[i].lbound = 1
            self._ictype.dims[i].ubound = shape[i]

    @property
    def strides(self):
        if not self.isAllocated():
            return

        strides = []
        for i in range(self.ndims):
            strides.append(self._ictype.dims[i].stride * ctypes.sizeof(self.elem))

        return tuple(strides)   

    @strides.setter
    def strides(self, shape):
        strides = []
        for i in range(self.ndims):
            strides.append(self._ictype.dims[i].ubound - self._ictype.dims[i].lbound + 1)    
            
        sumstrides = 0
        for i in range(self.ndims):
            self._ictype.dims[i].stride = int(np.product(strides[:i]))
            sumstrides = sumstrides + self._ictype.dims[i].stride

        self._ictype.offset = -sumstrides
        self._ictype.span = ctypes.sizeof(self.elem)
                
    @property
    def size(self):
        if not self.isAllocated():
            return
            
        return np.product(self.shape)  

    @property
    def base_addr(self):
        return self._ictype.base_addr

    @base_addr.setter
    def base_addr(self, addr):
        self._ictype.base_addr = addr


    def get(self):
        return self._ictype


    def set(self, addr, shape):
        if not self.isAllocated():
            self.allocate()

        self.base_addr = addr
        self.shape = shape
        self.strides = shape

        self.set_dtype15()
 
    def set_dtype15(self):
        dtype = self._ictype.dtype
        
        dtype.elem_len = ctypes.sizeof(self.elem)
        dtype.version = 0
        dtype.rank = self.ndims
        dtype.attribute = 0
        ftype = self.get_ftype()
        
        self._ictype.type = ftype

    def get_ftype(self):

        elem = self.elem()

        if hasattr(elem,'_type_'):
            if callable(elem._type_):
                elem = elem._type_()

        if any([isinstance(elem, c) for c in [ctypes.c_int, ctypes.c_int32, ctypes.c_int64]]):
            ftype = _BT_INTEGER
        elif any([isinstance(elem, c) for c in [ctypes.c_float, ctypes.c_double]]):
            ftype = _BT_REAL
        elif isinstance(elem, ctypes.c_bool):
            ftype = _BT_LOGICAL
        elif any([isinstance(elem, c) for c in [ctypes.c_char, ctypes.c_char_p]]): 
            ftype = _BT_CHARACTER
        elif isinstance(elem, ctypes.Structure):
            if isinstance(elem, _complex):
                ftype = _BT_COMPLEX
            else: 
                ftype = _BT_DERIVED
        else:
            raise ValueError("Cant match dtype, got " + str(self.elem))
        return ftype