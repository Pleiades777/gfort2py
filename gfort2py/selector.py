# SPDX-License-Identifier: GPL-2.0+
from .cmplx import fComplex, fParamComplex
from .arrays import fExplicitArray, fParamArray, fAssumedShape, fAssumedSize
from .strings import fStr
from .var import fVar, fParam


def _selectVar(obj):
    x = None

    if 'param' in obj:
        # Parameter strings are just be fParams
        if obj['param']['pytype'] == 'complex':
            x = fParamComplex
        elif obj['param']['array']:
            if obj['param']['array']:
                x = fParamArray
        else:
            x = fParam
    elif 'var' in obj:
        if obj['var']['pytype'] == 'str':
            x = fStr
        elif obj['var']['pytype'] == 'complex':
            x = fComplex
        elif 'dt' in obj['var'] and obj['var']['dt']:
            from .types import fDerivedType
            x = fDerivedType
        elif 'array' in obj['var']:
            array = obj['var']['array']['atype']
            if array == 'explicit':
                x = fExplicitArray
            elif any(array in i for i in ['assumed_shape','alloc','pointer']):
                x = fAssumedShape
            elif array == 'assumed_size':
                x = fAssumedSize
        elif 'is_func' in obj['var'] and obj['var']['is_func']:
            from .functions import fFuncPtr
            x = fFuncPtr
        else:
            x = fVar

    return x
