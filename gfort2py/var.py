from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np

# Hacky, yes
__builtin__.quad = np.longdouble


class NotInLib(Exception):
    pass


class fVar(object):

    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self.ctype=self.var['ctype']
        self.pytype=self.var['pytype']
        self._ctype = self.ctype_def()
        #self._ctype_f = self.ctype_def_func()
        self._pytype = self.pytype_def()
        if self.pytype == 'quad':
            self.pytype = np.longdouble

        # if true for things that are fortran things
        self._fortran = True
        
        # True if its a function argument
        self._func_arg=False
        
        #True if struct member
        self._dt_arg=False
        
        self.TEST_FLAG=TEST_FLAG

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        return self.ctype_def()(value)
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        x,_=self.ctype_def_func()
        return x(self.ctype_def()(value)),None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._pytype(value.value)
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        if hasattr(value,'contents'):
            return self._pytype(value.contents.value)
        else:
            return self._pytype(value.value)

    def pytype_def(self):
        return getattr(__builtin__, self.pytype)

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return getattr(ctypes, self.ctype)

    def ctype_def_func(self,pointer=False):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        f = ctypes.POINTER(self.ctype_def())
        if pointer:
            f = ctypes.POINTER(f)

        return f,None
        
    def py_to_ctype_p(self,value):
        """
        The ctype represnation suitable for function arguments wanting a pointer
        """

        return ctypes.POINTER(self.ctype_def())(self.py_to_ctype(value))
        

    def set_mod(self, value):
        """
        Set a module level variable
        """
        r = self._get_from_lib()
        r.value = self._pytype(value)

    def get(self,copy=True):
        """
        Get a module level variable
        """
        r = self._get_from_lib()
        if copy:
            res = self.ctype_to_py(r)
        else:
            if hasattr(r,'contents'):
                res =r.contents
            else:
                res = r
        
        return res

    def _get_from_lib(self):
        if 'mangled_name' in self.__dict__ and '_lib' in self.__dict__:
            return self._ctype.in_dll(self._lib, self.mangled_name)
        raise NotInLib   
        

    def _get_var_by_iter(self, value, size=-1):
        """ Gets a variable where we have to iterate to get multiple elements"""
        base_address = ctypes.addressof(value)
        return self._get_var_from_address(base_address, size=size)

    def _get_var_from_address(self, ctype_address, size=-1):
        out = []
        i = 0
        sof = ctypes.sizeof(self._ctype)
        while True:
            if i == size:
                break
            x = self._ctype.from_address(ctype_address + i * sof)
            if x.value == b'\x00':
                break
            else:
                out.append(x.value)
            i = i + 1
        return out

    def _set_var_from_iter(self, res, value, size=99999):
        base_address = ctypes.addressof(res)
        self._set_var_from_address(base_address, value, size)

    def _set_var_from_address(self, ctype_address, value, size=99999):
        for j in range(min(len(value), size)):
            offset = ctype_address + j * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = value[j]

    def _pname(self):
        return str(self.name) + " <" + str(self.pytype) + ">"

    def __str__(self):
        try:
            return str(self.get(), name)
        except NotInLib:
            return None

    def __repr__(self):
        s=''
        try:
            s=str(self.get()) + " <" + str(self.pytype) + ">"
        except (ValueError,AttributeError,NotInLib):
            # Skip for things that aren't in the module (function arg)
            s=" <" + str(self.pytype) + ">"
        return s
    
    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]
            
        if '_func_arg' in self.__dict__:
            if self._func_arg:
                return   
        
        if '_dt_arg' in self.__dict__:
            if self._dt_arg:
                return 
                 
        try:
            return getattr(self.get(), name)
        except:
            return None

    #Stuff to call the result of self.get() (a python object int/str etc)

    def __add__(self, other):
        return getattr(self.get(), '__add__')(other)

    def __sub__(self, other):
        return getattr(self.get(), '__sub__')(other)

    def __mul__(self, other):
        return getattr(self.get(), '__mul__')(other)

    def __matmul__(self,other):
        return getattr(self.get(), '__matmul__')(other)

    def __truediv__(self, other):
        return getattr(self.get(), '__truediv__')(other)
        
    def __floordiv__(self,other):
        return getattr(self.get(), '__floordiv__')(other)

    def __pow__(self, other, modulo=None):
        return getattr(self.get(), '__pow__')(other,modulo)

    def __mod__(self,other):
        return getattr(self.get(), '__mod__')(other)        
        
    def __lshift__(self,other):
        return getattr(self.get(), '__lshift__')(other)        

    def __rshift__(self,other):
        return getattr(self.get(), '__rshift__')(other)

    def __and__(self,other):
        return getattr(self.get(), '__and__')(other)
        
    def __xor__(self,other):
        return getattr(self.get(), '__xor__')(other)
        
    def __or__(self,other):
        return getattr(self.get(), '__or__')(other)
        
    def __radd__(self, other):
        return getattr(self.get(), '__radd__')(other)

    def __rsub__(self, other):
        return getattr(self.get(), '__rsub__')(other)

    def __rmul__(self, other):
        return getattr(self.get(), '__rmul__')(other)

    def __rmatmul__(self,other):
        return getattr(self.get(), '__rmatmul__')(other)

    def __rtruediv__(self, other):
        return getattr(self.get(), '__rtruediv__')(other)
        
    def __rfloordiv__(self,other):
        return getattr(self.get(), '__rfloordiv__')(other)

    def __rpow__(self, other):
        return getattr(self.get(), '__rpow__')(other)

    def __rmod__(self,other):
        return getattr(self.get(), '__rmod__')(other)        
        
    def __rlshift__(self,other):
        return getattr(self.get(), '__rlshift__')(other)        

    def __rrshift__(self,other):
        return getattr(self.get(), '__rrshift__')(other)

    def __rand__(self,other):
        return getattr(self.get(), '__rand__')(other)
        
    def __rxor__(self,other):
        return getattr(self.get(), '__rxor__')(other)
        
    def __ror__(self,other):
        return getattr(self.get(), '__ror__')(other)

    def __iadd__(self, other):
        self.set_mod(self.get() + other)
        return self.get()

    def __isub__(self, other):
        self.set_mod(self.get() - other)
        return self.get()

    def __imul__(self, other):
        self.set_mod(self.get() * other)
        return self.get()

    def __itruediv__(self, other):
        self.set_mod(self.get() / other)
        return self.get()

    def __ipow__(self, other, modulo=None):
        x = self.get()**other
        if modulo:
            x = x % modulo
        self.set_mod(x)
        return self.get()

    def __eq__(self, other):
        return getattr(self.get(), '__eq__')(other)

    def __neq__(self, other):
        return getattr(self.get(), '__new__')(other)

    def __lt__(self, other):
        return getattr(self.get(), '__lt__')(other)

    def __le__(self, other):
        return getattr(self.get(), '__le__')(other)

    def __gt__(self, other):
        return getattr(self.get(), '__gt__')(other)

    def __ge__(self, other):
        return getattr(self.get(), '__ge__')(other)
        
    def __format__(self, other):
        return getattr(self.get(), '__format__')(other)
  
    def __bytes__(self):
        return getattr(self.get(), '__bytes__')()  
        
    def __bool__(self):
        return getattr(self.get(), '__bool__')()
   
    def __len__(self):
        return getattr(self.get(), '__len__')()
 
    def __length_hint__(self):
        return getattr(self.get(), '__length_hint__')()       
        
    def __dir__(self):
        return list(self.__dict__.keys()) + list(dir(self.get()))


class fParam(fVar):
    def __init__(self, lib, obj,TEST_FLAG=False):
        self.__dict__.update(obj)
        self._lib = lib
        self.TEST_FLAG=TEST_FLAG
        self.value = self.param['value']
        self.pytype = self.param['pytype']
        self._pytype = self.pytype_def()

    def set_mod(self, value):
        """
        Cant set a parameter
        """
        raise ValueError("Can't alter a parameter")

    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them 
        from the shared lib.
        """
        return self._pytype(self.value)
