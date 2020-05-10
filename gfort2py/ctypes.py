# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function
import ctypes

def get_complex(c):

    class _complex(ctypes.Structure):
        _fields_ = [('real', c),
                    ('imag', c)]

    return _complex