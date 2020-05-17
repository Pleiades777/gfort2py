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


class fExplicitArray():
    """ Wrapper for gfortrans explicit arrays

    These inclide:
        dimension(5)
        dimension(N)

    Arguments:
        object {dict} -- Dictionary containg the objects properties
    """
    def __init__(self, obj):
        self.var = obj['var']
        if 'mangled_name' in obj:
            self.mangled_name = obj['mangled_name']
        self._array = True
        if 'name' in obj:
            self.name = obj['name']

        if 'array' in self.var:
            self.array = obj['var']['array']

        self.ctype = self.var['ctype']
        self.pytype = self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype == 'bool':
            self.pytype = int
            self.ctype = 'c_int32'
        else:
            self.pytype = getattr(__builtin__, self.pytype)

        self.ctype = getattr(ctypes, self.ctype)
        self._sof_ctype = ctypes.sizeof(self.ctype)

        if self.pytype == int:
            self._dtype = 'int' + str(8 * ctypes.sizeof(self.ctype))
        elif self.pytype == float:
            self._dtype = 'float' + str(8 * ctypes.sizeof(self.ctype))
        else:
            raise NotImplementedError("Type not supported ", self.pytype)

        self._ndims = int(self.array['ndim'])

        if self.size() > 0:
            self.ctype = self.ctype * self.size()

        self._shape = self.shape()
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
            'typestr': self._dtype,
            'shape': self._shape,
            'version': 3,
            'strides': self.strides()
        }

        class numpy_holder():
            pass

        holder = numpy_holder()
        holder.__array_interface__ = buff

        arr = np.asfortranarray(holder)
        remove_ownership(arr)

        return arr

    def strides(self):
        """Compute the array strides

        Returns:
            [tuple(ints)] -- Get the strides of the array in bytes
        """
        if self._shape == -1:
            return None

        strides = [self._sof_ctype]
        for i in self._shape[:-1]:
            strides.append(strides[-1] * i)

        return tuple(strides)

    def shape(self):
        """ Compute the shape of an array

        Returns:
            [tuple(ints)] -- Tuple of array shape in python form
        """
        if 'shape' not in self.array or len(
                self.array['shape']) / self._ndims != 2:
            return -1

        shape = []
        for l, u in zip(self.array['shape'][0::2], self.array['shape'][1::2]):
            shape.append(u - l + 1)
        return tuple(shape)

    def size(self):
        """ Size of array

        Returns:
            [int] -- Total number of elements in the array
        """
        return np.product(self.shape())

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
        if v.ndim != self._ndims:
            raise AttributeError("Bad ndims for array")

        if v.shape != self._shape and not self._shape == -1:
            raise AttributeError("Bad shape for array")

        self._save_value(v)
        v_addr = self._value.ctypes.data

        # print(v.shape,v.itemsize,v.size*v.itemsize,self._dtype,self.sizeof(),c)
        ctypes.memmove(ctypes.addressof(c), v_addr, self.sizeof())

    # def set(self, value):
    #    self._set(self.in_dll(), value)

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
        size = np.size(value)

        if self.shape() == -1:
            self._shape = value.shape
            self.ctype = self.ctype * size

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

    def sizeof(self):
        """ Compute total size of array

        Returns:
            [int] -- Size of array in bytes
        """
        return ctypes.sizeof(self.ctype)

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self._dtype))
        remove_ownership(self._value)


class fDummyArray(object):
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
        self.var = obj['var']
        if 'mangled_name' in obj:
            self.mangled_name = obj['mangled_name']
        self._array = self.var['array']
        if 'name' in obj:
            self.name = obj['name']

        self.ndim = int(self._array['ndim'])

        self.ctype_elem = self.var['ctype']
        self.pytype = self.var['pytype']

        if self.pytype == 'quad':
            self.pytype = np.longdouble
        elif self.pytype == 'bool':
            self.pytype = int
            self.ctype_elem = 'c_int32'
        else:
            self.pytype = getattr(__builtin__, self.pytype)

        self.ctype_elem = getattr(ctypes, self.ctype_elem)
        self._sof_ctype = ctypes.sizeof(self.ctype_elem)

        if self.pytype == int:
            self._dtype = 'int' + str(8 * ctypes.sizeof(self.ctype_elem))
        elif self.pytype == float:
            self._dtype = 'float' + str(8 * ctypes.sizeof(self.ctype_elem))
        else:
            raise NotImplementedError("Type not supported yet ", self.pytype)

        self.array_desc = arrayDescriptor(self.ndim, elem=self.ctype_elem)
        self.ctype = self.array_desc.ctype
        self._value = None

    def from_address(self, addr):
        self.array_desc.from_address(addr)
        buff = {
            'data': (self.array_desc.base_addr,
                     False),
            'typestr': self._dtype,
            'shape': self.array_desc.shape,
            'version': 3,
            'strides': self.array_desc.strides
        }

        class numpy_holder():
            pass

        holder = numpy_holder()
        holder.__array_interface__ = buff
        arr = np.asfortranarray(holder)
        remove_ownership(arr)

        return arr

    def set_from_address(self, addr, value):
        self._save_value(value)
        self.array_desc.set(addr, np.shape(self._value))

    def in_dll(self, lib):
        self.array_desc.in_dll(lib, self.mangled_name)
        if not self.array_desc.isAllocated():
            raise AllocationError("Array not allocated yet")
        return self.from_address(self.array_desc.addressof())

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self._dtype))
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
