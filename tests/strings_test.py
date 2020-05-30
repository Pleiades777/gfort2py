# SPDX-License-Identifier: GPL-2.0+

import os, sys

os.environ["_GFORT2PY_TEST_FLAG"] = "1"

import numpy as np
import gfort2py as gf

import unittest as unittest
    
import subprocess
import numpy.testing as np_test

from contextlib import contextmanager
from io import StringIO, BytesIO

#Decreases recursion depth to make debugging easier
# sys.setrecursionlimit(10)

SO = './tests/strings.so'
MOD = './tests/strings.mod'

x=gf.fFort(SO,MOD,rerun=True)



@contextmanager
def captured_output():
    """
    For use when we need to grab the stdout/stderr from fortran (but only in testing)
    Use as:
    with captured_output() as (out,err):
        func()
    output=out.getvalue().strip()
    error=err.getvalue().strip()
    """
    new_out, new_err = StringIO(),StringIO()
    old_out,old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

class TestStringMethods(unittest.TestCase):
    def test_a_str(self):
        v='123456798 '
        x.a_str=v
        self.assertEqual(x.a_str,v)
        
    def test_a_str_bad_length(self):
        v='132456789kjhgjhf'
        x.a_str=v
        self.assertEqual(x.a_str,v[0:10])
        
    def test_sub_str_in_explicit(self):
        v='1324567980'
        with captured_output() as (out,err):
            y=x.sub_str_in_explicit(v)
        output=out.getvalue().strip()
        self.assertEqual(output,v)	
        
    def test_sub_str_in_implicit(self):
        v='123456789'
        with captured_output() as (out,err):
            y=x.sub_str_in_implicit(v)
        output=out.getvalue().strip()	
        self.assertEqual(output,v)	
    
    def test_sub_str_multi(self):
        v=5
        u='123456789'
        w=4
        with captured_output() as (out,err):
            y=x.sub_str_multi(v,u,w)
        output=out.getvalue().strip()	
        self.assertEqual(output,str(v+w)+' '+u)	
        
    def test_sub_str_p(self):
        with captured_output() as (out,err):
            y=x.sub_str_p('abcdef')
        output=out.getvalue().strip()
        self.assertEqual(output,'abcdef')
        self.assertEqual(y.args['zzz'],'xyzxyz')
    
    def test_func_ret_str(self):
        y=x.func_ret_str('abcde')
        self.assertEqual(y.result,'Abcde')   
        
    @unittest.skip("Skipping")	
    # We need to call a func on the argument before passing it to func_str_int_len
    def test_func_str_int_len(self):
        with captured_output() as (out,err):
            y=x.func_str_int_len(10)
            
        self.assertEqual(out,'10')   
    
    @unittest.skip("Skipping")
    def test_str_alloc(self):
        self.assertEqual(x.str_alloc, '') # Empty at start

        x.str_alloc = 'abcdefghijklmnop'
        self.assertEqual(x.str_alloc, 'abcdefghijklmnop')
        y = x.check_str_alloc(1)
        self.assertEqual(y.result,True)
        
        x.str_alloc = '12345678        ' # Need to empty the space afterwards
        self.assertEqual(x.str_alloc,  '12345678        ')    
        y = x.check_str_alloc(2)    
        self.assertEqual(y.result,True)

    def test_b_str10_exp_1d(self):
        x.clear_strs()
        empty = np.array([b' '*10]*5, dtype='|S10')

        np_test.assert_array_equal(x.b_str10_exp_1d, empty)

        test = 'abcdefghik'
        x.sub_set_b_str10_exp_1d(test)
        y=x.sub_check_b_str10_exp_1d(test)
        self.assertEqual(y.result,True)
        testArr=np.array([test]*5, dtype='|S10')
        np_test.assert_array_equal(x.b_str10_exp_1d, testArr)

    def test_b_str10_exp_1d_bad_len(self):
        x.clear_strs()
        empty = np.array([b' '*10]*5, dtype='|S10')

        np_test.assert_array_equal(x.b_str10_exp_1d, empty)

        test = 'abcdefg'
        x.sub_set_b_str10_exp_1d(test)
        y=x.sub_check_b_str10_exp_1d(test)
        self.assertEqual(y.result,True)
        testArr=np.array([test]*5, dtype='|S10')
        np_test.assert_array_equal(x.b_str10_exp_1d, testArr)

    def test_b_str10_exp_2d(self):
        x.clear_strs()
        empty = np.array([b' '*10]*25, dtype='|S10').reshape(5,5)

        np_test.assert_array_equal(x.b_str10_exp_2d, empty)   

        test = 'abcdefghik'
        x.sub_set_b_str10_exp_2d(test)
        y=x.sub_check_b_str10_exp_2d(test)
        self.assertEqual(y.result,True)
        testArr=np.array([test]*25, dtype='|S10').reshape(5,5)
        np_test.assert_array_equal(x.b_str10_exp_2d, testArr)



    def test_c_str10_alloc_1d(self):
        x.sub_alloc_strs()

        test = 'abcdefghik'
        v = np.array([test]*5, dtype='|S10')

        x.c_str10_alloc_1d = v
        np_test.assert_array_equal(x.c_str10_alloc_1d, v)

        y = x.sub_check_c_str10_alloc_1d(test)
        self.assertEqual(y.result,True)

        test2 = 'asdfghj'
        v = np.array([test2]*5, dtype='|S10')
        x.sub_set_c_str10_alloc_1d(test2)
        np_test.assert_array_equal(x.c_str10_alloc_1d, v)


    def test_c_str10_alloc_2d(self):
        x.sub_alloc_strs()

        test = 'abcdefghik'
        v = np.array([test]*25, dtype='|S10').reshape(5,5)

        x.c_str10_alloc_2d = v
        np_test.assert_array_equal(x.c_str10_alloc_2d, v)

        y = x.sub_check_c_str10_alloc_2d(test)
        self.assertEqual(y.result,True)        

        test2 = 'asdfghj'
        v = np.array([test2]*25, dtype='|S10').reshape(5,5)
        x.sub_set_c_str10_alloc_2d(test2)
        np_test.assert_array_equal(x.c_str10_alloc_2d, v)


    def test_sub_strStar_explict_N(self):

        n = 8
        v = np.zeros(n,dtype='|S11')

        y = x.sub_strStar_explict_N(n, v)

        np_test.assert_array_equal(y.args['s'],np.array([b'zxcvbnm    ', b'zxcvbnm    ', b'zxcvbnm    ', b'zxcvbnm    ',
                 b'zxcvbnm    ', b'zxcvbnm    ', b'zxcvbnm    ', b'zxcvbnm    ']))
    
if __name__ == '__main__':
    unittest.main() 
