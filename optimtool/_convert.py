# Copyright (c) 2021 linjing-lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sympy as sp
import numpy as np
from ._typing import FuncArray, ArgArray, PointArray, SympyMutableDenseMatrix, NDArray

def f2m(funcs: FuncArray) -> SympyMutableDenseMatrix:
    '''
    Parameters
    ----------
    funcs : FuncArray
        目标函数

    Returns
    -------
    funcs : SympyMutableDenseMatrix
        目标函数

    '''
    # convert funcs
    return sp.Matrix(funcs) if isinstance(funcs, (list, tuple)) else sp.Matrix([funcs])

def a2m(args: ArgArray) -> SympyMutableDenseMatrix:
    '''
    Parameters
    ----------
    args : ArgArray
        参数

    Returns
    -------
    args : SympyMutableDenseMatrix
        参数

    '''
    # convert args
    return sp.Matrix(args) if isinstance(args, (list, tuple)) else sp.Matrix([args])

def p2t(x_0: PointArray) -> PointArray:
    '''
    Parameters
    ----------
    x_0 : PointArray
        参数

    Returns
    -------
    x_0 : PointArray
        参数

    '''
    # convert x_0
    return (x_0,) if not isinstance(x_0, (list, tuple)) else x_0

def h2h(hessian: NDArray, pk: int=1) -> NDArray:
    '''
    Parameters
    ----------
    hessian : numpy.array
        未修正的海瑟矩阵值
        
    pk : int
        常数
        

    Returns
    -------
    numpy.array
        修正后的海瑟矩阵
        
    '''
    l = hessian.shape[0] # hessian.shape = (l, l)
    while 1:
        rank = np.linalg.matrix_rank(hessian)
        if rank == l:
            break
        else:
            hessian = hessian + pk * np.identity(l)
            pk += 1
    return hessian
__all__ = [f2m, a2m, p2t, h2h]