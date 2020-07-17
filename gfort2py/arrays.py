# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

from .fnumpy import remove_ownership
from .errors import AllocationError, IgnoreReturnError

from .descriptors import arrayInterfaceDescriptor, arrayExplicitDescriptor
from .quad import bytes2quad, quad2bytes

class BadFortranArray(Exception):
    pass


class fArray():
    def __init__(self, obj):
        self.var = obj['var']
        self.array_desc = None
        if 'mangled_name' in obj:
            self.mangled_name = obj['mangled_name']

        self._array = True
        if 'name' in obj:
            self.name = obj['name']

        if 'array' in self.var:
            self.array = obj['var']['array'] 

        self.ctype_elem = getattr(ctypes, self.var['ctype'])

        if self.var['pytype'] == 'quad':
            self.pytype = 'quad'
            self.ctype_elem = ctypes.c_ubyte * 16
        elif self.var['pytype'] == 'bool':
            self.pytype = int
            self.ctype_elem = ctypes.c_int32
        elif self.var['pytype'] =='str':
            self.pytype = str
            self.ctype_elem = ctypes.c_char
        else:
            self.pytype = getattr(__builtin__, self.var['pytype'])

        self.length = -1
        if 'length' in self.var:
            self.length = self.var['length']

        self.ndim = int(self.array['ndim'])

        if 'shape' not in self.array:
            self._shape = -1
        else:
            if len(self.array['shape']) / self.ndim != 2:
                self._shape = -1
            else:
                self._shape = self.array['shape']

        self._value = None

    def from_address(self, addr):
        """ Create a numpy array from address

        Arguments:
            addr {[int]} -- Address for the start of an array

        Returns:
            [array] -- Returns a numpy array from the address addr
        """

        if self.pytype == 'quad':
            return self
        else:
            buff = {
                'data': (addr,
                        False),
                'typestr': self.dtype,
                'shape': self.shape,
                'version': 3,
                'strides': self.strides
            }

            class numpy_holder():
                pass

            holder = numpy_holder()
            holder.__array_interface__ = buff
            # print(addr,self.dtype,self.shape,self.strides)
            arr = np.asfortranarray(holder)
            remove_ownership(arr)
            return arr


    @property
    def dtype(self):
        if self.pytype == int:
            return 'int' + str(8 * ctypes.sizeof(self.array_desc.elem))
        elif self.pytype == float:
            return 'float' + str(8 * ctypes.sizeof(self.array_desc.elem))
        elif self.pytype == str:
            return '|S' + str(ctypes.sizeof(self.array_desc.elem))
        elif self.pytype == 'quad':
            return 'uint8'
        else:
            raise NotImplementedError("Type not supported ", self.pytype)

    @property
    def strides(self):
        """Compute the array strides

        Returns:
            [tuple(ints)] -- Get the strides of the array in bytes
        """
        return self.array_desc.strides

    @property
    def shape(self):
        """ Compute the shape of an array

        Returns:
            [tuple(ints)] -- Tuple of array shape in python form
        """
        return self.array_desc.shape

    @property
    def size(self):
        """ Size of array

        Returns:
            [int] -- Total number of elements in the array
        """
        return self.array_desc.size

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self.dtype))
        remove_ownership(self._value)


    def sizeof(self):
        """ Compute total size of array

        Returns:
            [int] -- Size of array in bytes
        """
        return ctypes.sizeof(self.ctype)

    def from_len(self, string, length):
        if hasattr(length,'value'):
            length= length.value

        self.length = length
        self.array_desc.length = length
        return self.from_func(string)

    @property
    def ctype(self):
        return self.array_desc.ctype

    # def set_from_address(self, addr, value):
    #     self._save_value(value)
    #     self.array_desc.set(addr, np.shape(self._value))

    def in_dll(self, lib):
        self.array_desc.in_dll(lib, self.mangled_name)
        if not self.array_desc.isAllocated():
            raise AllocationError("Array not allocated yet")
        return self.from_address(self.array_desc.addressof())

    def _save_value(self, value):
        if self.pytype == 'quad':
            v = _map_multid(value, quad2bytes)
            self._value = np.asfortranarray(v,dtype=self.dtype)
            self.set_length(self._value)
        else:
            self.set_length(value)
            self._value = np.asfortranarray(value,dtype=self.dtype)
            remove_ownership(self._value)


    def set_in_dll(self, lib, value):
        self.array_desc.in_dll(lib, self.mangled_name)
        self._save_value(value)
        self.set_length(value)
        self.array_desc.set(self._value.ctypes.data, np.shape(self._value))

    def set_length(self, value):
        if value.dtype.char == 'S' and self.pytype == str:
            if self.array_desc.length == -1:
                self.array_desc.length = value.dtype.itemsize
            else:
                if self.array_desc.length < value.dtype.itemsize:
                    raise AttributeError("String elements of array are too long")


    def from_param(self, value):
        if self.array['atype'] == 'alloc' or self.array['atype'] == 'pointer':
            if value is not None:
                self._save_value(value)
                self.array_desc.set(self._value.ctypes.data, np.shape(self._value))

            self.set_length(value)
            return self.array_desc.pointer()
        else:
            self._save_value(value)
            self._shape = np.shape(self._value)
            self.array_desc._shape = np.shape(self._value)
            self.array_desc.set(self._value.ctypes.data, np.shape(self._value))
            return self.array_desc.from_param()

    def __getitem__(self, key):
        val = self.array_desc.__getitem__(key)

        if self.pytype == 'quad':
            return bytes2quad(val)
        else:
            return val

    def __setitem__(self, key, value):
        if self.pytype == 'quad':
            value = quad2bytes(value)

        self.array_desc.__setitem__(key, value)    


class fExplicitArray(fArray):
    def __init__(self, obj):
        super().__init__(obj)

        self.array_desc = arrayExplicitDescriptor(self.ndim, elem=self.ctype_elem,
                            length=self.length, shape=self._shape)

    def from_func(self, pointer):
        x = pointer
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        if x is None:
            return None

        try:
            return self.from_address(ctypes.addressof(x))
        except AttributeError:
            raise IgnoreReturnError

class fDummyArray(fArray):
    """Wrapper for gfortrans dummy arrays

    These inclide:
        dimension(:)
        dimension(:), allocatable
        dimension(:), pointer
        dimension(:), target

    Arguments:
        object {dict} -- Dictionary containg the objects properties
    """
    def __init__(self, obj):
        super().__init__(obj)

        self.array_desc = arrayInterfaceDescriptor(self.ndim, elem=self.ctype_elem)

    def from_func(self, pointer):
        x = pointer
        if hasattr(pointer, 'contents'):
            if hasattr(pointer.contents, 'contents'):
                x = pointer.contents.contents
            else:
                x = pointer.contents

        if x is None:
            return None

        try:
            return self.from_address(ctypes.addressof(x))
        except AttributeError:
            raise IgnoreReturnError    

    def from_address(self, addr):
        self.array_desc.from_address(addr)
        return super().from_address(self.array_desc.base_addr)



class fAssumedShape(fDummyArray):
    """ Wrapper for gfortran's assumed shape arrays

    These include:
        dimension(:)

    Arguments:
        object {dict} -- Dictionary containg the objects properties
    """

    def from_param(self, value):
        if value is not None:
            self._save_value(value)
            self.array_desc.set(self._value.ctypes.data, np.shape(self._value))

        return self.array_desc.pointer()



class fAssumedSize(fExplicitArray):
    """ Wrapper for gfortrans assumed size arrays

    These include:
        dimension(*)

    Arguments:
        object {dict} -- Dictionary containg the objects properties
    """

    # Only difference between this and an fExplicitArray is we don't know the shape.
    # We just pass the pointer to first element

    def from_param(self, value):
        """ Convert input array into form suitable for passing to a function

        Arguments:
            value {[array]} -- Numpy array to copy

        Returns:
            [ctype] -- ctype representation of value
        """
        #self.array_desc._length = value.itemsize
        self.array_desc._shape = value.shape
        return super().from_param(value)

    @property
    def shape(self):
        if hasattr(self,'_shape'):
            return self._shape
        else:
            return super().shape



class fParamArray():
    """ Wrapper for gfortran's parameter arrays

    These include:
        dimension(5),parameter

    Arguments:
        object {dict} -- Dictionary containg the objects properties
    """
    def __init__(self, obj):
        self.param = obj['param']
        self.pytype = self.param['pytype']
        self.pytype = getattr(__builtin__, self.pytype)
        self.value = np.array(
            self.param['value'],
            dtype=self.pytype,
            order='F')

    def set_in_dll(self, lib, value):
        """
        Can't set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def in_dll(self, lib):
        """
        A parameters value is stored in the dict, as we can't access them
        from the shared lib.
        """
        return self.value


class fStrLenArray():
    # Handles the hidden string length functions need
    def __init__(self):
        self.ctype = ctypes.c_int64

    def from_param(self, value):
        return self.ctype(value.dtype.itemsize)

    def from_func(self, pointer):
        raise IgnoreReturnError


def _map_multid(mylist,mymap):
    res = []
    for i in mylist:
        if isinstance(i,list):
            res.append(_map_multid(i,mymap))
        else:
            res.append(mymap(i))
    return res