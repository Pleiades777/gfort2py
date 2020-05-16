# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes
import sys
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


class fExplicitArray(object):
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

    def from_address(self, addr):
        self._base_holder = addr  # Make sure to hold onto the object
        buff = {
            'data': (self._base_holder,
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
        if self._shape == -1:
            return None

        strides = [self._sof_ctype]
        for i in self._shape[:-1]:
            strides.append(strides[-1] * i)

        return tuple(strides)

    def shape(self):
        if 'shape' not in self.array or len(
                self.array['shape']) / self._ndims != 2:
            return -1

        shape = []
        for l, u in zip(self.array['shape'][0::2], self.array['shape'][1::2]):
            shape.append(u - l + 1)
        return tuple(shape)

    def size(self):
        return np.product(self.shape())

    def set_from_address(self, addr, value):
        ctype = self.ctype.from_address(addr)
        self._set(ctype, value)

    def _set(self, c, v):
        if v.ndim != self._ndims:
            raise AttributeError("Bad ndims for array")

        if v.shape != self._shape and not self._shape == -1:
            raise AttributeError("Bad shape for array")

        self._value = v

        self._value = np.asfortranarray(self._value .astype(self._dtype))
        v_addr = self._value.ctypes.data

        # print(v.shape,v.itemsize,v.size*v.itemsize,self._dtype,self.sizeof(),c)
        ctypes.memmove(ctypes.addressof(c), v_addr, self.sizeof())
        remove_ownership(self._value)

    # def set(self, value):
    #    self._set(self.in_dll(), value)

    def in_dll(self, lib):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        return self.from_address(addr)

    def set_in_dll(self, lib, value):
        addr = ctypes.addressof(self.ctype.in_dll(lib, self.mangled_name))
        self.set_from_address(addr, value)

    def from_param(self, value):
        size = np.size(value)

        if self.shape() == -1:
            self._shape = value.shape
            self.ctype = self.ctype * size

        self._safe_ctype = self.ctype()
        self._set(self._safe_ctype, value)
        return self._safe_ctype

    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer))

    def sizeof(self):
        return ctypes.sizeof(self.ctype)


class fDummyArray(object):
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

        self._array_desc = arrayDescriptor(self.ndim, elem=self.ctype_elem)
        self.ctype = self._array_desc.ctype


    def from_address(self, addr):
        v = self._array_desc.from_address(addr)
        buff = {
            'data': (self._array_desc.base_addr,
                     False),
            'typestr': self._dtype,
            'shape': self._array_desc.shape,
            'version': 3,
            'strides': self._array_desc.strides
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
        self._array_desc.set(addr, np.shape(self._value))

    def in_dll(self, lib):
        self._array_desc.in_dll(lib, self.mangled_name)
        if not self._array_desc.isAllocated():
            raise AllocationError("Array not allocated yet")
        return self.from_address(self._array_desc.addressof())

    def _save_value(self, value):
        self._value = np.asfortranarray(value.astype(self._dtype))
        remove_ownership(self._value)

    def set_in_dll(self, lib, value):
        self._array_desc.in_dll(lib, self.mangled_name)
        self._save_value(value)
        self._array_desc.set(self._value.ctypes.data, np.shape(self._value))

    def from_param(self, value):
        if self._array['atype'] == 'alloc' or self._array['atype'] == 'pointer':
            if value is not None:
                self._save_value(value)
                self._array_desc.set(self._value.ctypes.data, np.shape(self._value))

            return self._array_desc.pointer()
        else:
            self._save_value(value)
            self._array_desc.set(self._value.ctypes.data, np.shape(self._value))
            return self._array_desc.from_param()

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

    def from_param(self, value):
        if value is not None:
            self._save_value(value)
            self._array_desc.set(self._value.ctypes.data, np.shape(self._value))

        return self._array_desc.pointer()

    def from_func(self, pointer):
        return self.from_address(ctypes.addressof(pointer.contents))


class fAssumedSize(fExplicitArray):
    # Only difference between this and an fExplicitArray is we don't know the shape.
    # We just pass the pointer to first element
    pass


class fParamArray(object):
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
        Cant set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def in_dll(self, lib):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return self.value
