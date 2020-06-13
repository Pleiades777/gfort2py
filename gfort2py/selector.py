# SPDX-License-Identifier: GPL-2.0+
from .cmplx import fComplex, fParamComplex
from .arrays import fExplicitArray, fParamArray, fAssumedShape, fAssumedSize, fQuadArray
from .strings import fStr
from .var import fVar, fParam


def _selectVar(obj):
    if 'param' in obj:
        # Parameter strings are just be fParams
        if obj['param']['pytype'] == 'complex':
            return fParamComplex
        elif obj['param']['array']:
            if obj['param']['array']:
                return fParamArray
        else:
            return fParam
    elif 'var' in obj:
        if 'dt' in obj['var'] and obj['var']['dt']:
            from .types import fDerivedType
            return fDerivedType
        elif 'array' in obj['var']:
            array = obj['var']['array']['atype']
            if array == 'explicit':
                if obj['var']['pytype'] == 'quad':
                    return fQuadArray
                else:
                    return fExplicitArray
            elif any(array in i for i in ['assumed_shape','alloc','pointer']):
                return fAssumedShape
            elif array == 'assumed_size':
                return fAssumedSize
        elif obj['var']['pytype'] == 'str':
            return fStr
        elif obj['var']['pytype'] == 'complex':
            return fComplex
        elif 'is_func' in obj['var'] and obj['var']['is_func']:
            from .functions import fFuncPtr
            return fFuncPtr
        else:
            return fVar

    return None
