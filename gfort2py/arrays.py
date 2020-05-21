# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

from .fnumpy import remove_ownership
from .errors import AllocationError, IgnoreReturnError

from .descriptors import arrayDescriptor


class BadFortranArray(Exception):
    pass



class fArray():
    def __init__(self, obj):
        self.var = obj['var']
        if 'mangled_name' in obj:
            self.mangled_name = obj['mangled_name']

        self._array = True
        if 'name' in obj:
            self.name = obj['name']

        if 'array' in self.var:
            self.array = obj['var']['array'] 

        self.ctype_elem = getattr(ctypes, self.var['ctype'])

        if self.var['pytype'] == 'quad':
            self.pytype = np.longdouble
        elif self.var['pytype'] == 'bool':
            self.pytype = int
            self.ctype_elem = ctypes.c_int32
        elif self.var['pytype'] =='str':
            self.pytype = str
            self.ctype_elem = ctypes.c_char
        else:
            self.pytype = getattr(__builtin__, self.var['pytype'])


        if 'length' in self.var:
            self.ctype_elem = self.ctype_elem * self.var['length']
        
        self._sof_ctype = ctypes.sizeof(self.ctype_elem)

        self.ndim = int(self.array['ndim'])

        self._value = None

    def from_address(self, addr):
        """ Create a numpy array from address

        Arguments:
            addr {[int]} -- Address for the start of an array

        Returns:
            [array] -- Returns a numpy array from the address addr
        """

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

        arr = np.asfortranarray(holder)
        remove_ownership(arr)

        return arr

    @property
    def dtype(self):
        if self.pytype == int:
            return 'int' + str(8 * self._sof_ctype)
        elif self.pytype == float:
            return 'float' + str(8 * self._sof_ctype)
        elif self.pytype == str:
            return '|S' + str(self._sof_ctype)
        else:
            raise NotImplementedError("Type not supported ", self.pytype)

    @property
    def strides(self):
        """Compute the array strides

        Returns:
            [tuple(ints)] -- Get the strides of the array in bytes
        """
        if self.shape == -1:
            return None

        strides = [self._sof_ctype]
        for i in self.shape[:-1]:
            strides.append(strides[-1] * i)

        return tuple(strides)

    @property
    def shape(self):
        """ Compute the shape of an array

        Returns:
            [tuple(ints)] -- Tuple of array shape in python form
        """
        if hasattr(self,'_shape'):
            return self._shape

        if 'shape' not in self.array or len(
                self.array['shape']) / self.ndim != 2:
            return -1

        shape = []
        for l, u in zip(self.array['shape'][0::2], self.array['shape'][1::2]):
            shape.append(u - l + 1)
        return tuple(shape)

    @property
    def size(self):
        """ Size of array

        Returns:
            [int] -- Total number of elements in the array
        """
        if 'length' in self.var:
            return np.product(self.shape * self.var['length'])
        elif hasattr(self,'_size'):
            return self._size
        else:
            return np.product(self.shape)

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self.dtype))
        remove_ownership(self._value)


    def sizeof(self):
        """ Compute total size of array

        Returns:
            [int] -- Size of array in bytes
        """
        return ctypes.sizeof(self.ctype)

class fExplicitArray(fArray):
    @property
    def ctype(self):
        if self.size > 0:
            return self.ctype_elem * self.size
        else:
            return None

    def set_from_address(self, addr, value):
        """ Set an array given by addr to value

        Arguments:
            addr {[int]} -- Destination address
            value {[array]} -- Source array
        """
        ctype = self.ctype.from_address(addr)
        self._set(ctype, value)

    def _set(self, c, v):
        """ Sets array given by a ctype to value

        Arguments:
            c {[ctype]} -- Instance of self.ctype to act as destination
            v {[array]} -- Source array

        Raises:
            AttributeError: If the number of dimensions or shape does not match
        """
        if v.ndim != self.ndim:
            raise AttributeError("Bad ndims for array")

        if v.shape != self.shape and not self.shape == -1:
            raise AttributeError("Bad shape for array")

        self._save_value(v)
        v_addr = self._value.ctypes.data

        # print(v.shape,v.itemsize,v.size*v.itemsize,self.dtype,self.sizeof(),c)
        ctypes.memmove(ctypes.addressof(c), v_addr, self.sizeof())

    def in_dll(self, lib):
        """Look up array in library

        Arguments:
            lib {[object]} -- ctypes loadlibraray() 

        Returns:
            [array] -- Numpy array given by self.mangled_name
        """
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        return self.from_address(addr)

    def set_in_dll(self, lib, value):
        """ Set up array in library

        Arguments:
            lib {[object]} -- ctypes loadlibraray() 
            value {[array]} -- Numpy array to copy

        """
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)

    def from_param(self, value):
        """ Convert input array into form suitable for passing to a function

        Arguments:
            value {[array]} -- Numpy array to copy

        Returns:
            [ctype] -- ctype representation of value
        """
        self._size = np.size(value)

        if self.shape == -1:
            self._shape = np.shape(value)


        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, value)
        return self._safe_ctype

    def from_func(self, pointer):
        """ Convert output from a function back into a numpy array

        Arguments:
            pointer {[ctype]} -- ctype representation of an array

        Returns:
            [array] -- Returned numpy array from function
        """
        return self.from_address(ctypes.addressof(pointer))


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

        self.array_desc = arrayDescriptor(self.ndim, elem=self.ctype_elem)

    @property
    def ctype(self):
        return self.array_desc.ctype

    def from_address(self, addr):
        self.array_desc.from_address(addr)
        return super().from_address(self.array_desc.base_addr)

    @property
    def strides(self):
        return self.array_desc.strides

    @property
    def shape(self):
        return self.array_desc.shape


    def set_from_address(self, addr, value):
        self._save_value(value)
        self.array_desc.set(addr, np.shape(self._value))

    def in_dll(self, lib):
        self.array_desc.in_dll(lib, self.mangled_name)
        if not self.array_desc.isAllocated():
            raise AllocationError("Array not allocated yet")
        return self.from_address(self.array_desc.addressof())

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self.dtype))
        remove_ownership(self._value)

    def set_in_dll(self, lib, value):
        self.array_desc.in_dll(lib, self.mangled_name)
        self._save_value(value)
        self.array_desc.set(self._value.ctypes.data, np.shape(self._value))

    def from_param(self, value):
        if self._array['atype'] == 'alloc' or self._array['atype'] == 'pointer':
            if value is not None:
                self._save_value(value)
                self.array_desc.set(self._value.ctypes.data, np.shape(self._value))

            return self.array_desc.pointer()
        else:
            self._save_value(value)
            self.array_desc.set(self._value.ctypes.data, np.shape(self._value))
            return self.array_desc.from_param()

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

    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer.contents))


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
        self._size = np.size(value)

        self._shape = value.shape

        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, value)
        return self._safe_ctype

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
