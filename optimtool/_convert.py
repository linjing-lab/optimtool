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

from .base import np, sp
from ._typing import ArgType, MulType, PowerType, AddType, FuncType, FuncArray, ArgArray, PointType, PointArray, SympyMutableDenseMatrix, NDArray

__all__ = ["f2m", "a2m", "p2t", "h2h"]

def f2m(funcs: FuncArray) -> SympyMutableDenseMatrix:
    '''
    :param funcs: FuncArray, objective function constructed with values of `symbols`.

    :return: objective function updated with `sp.Matrix()`.
    '''
    if isinstance(funcs, (SympyMutableDenseMatrix, AddType, PowerType, MulType, ArgType)): # add type of FuncType to support need!
        return sp.Matrix([funcs]) # prefer SympyMutableDenseMatrix to avoid enter into elif
    elif isinstance(funcs, (list, tuple)) and all(list(map(lambda x: isinstance(x, (AddType, PowerType, MulType, ArgType)), funcs))):
        return sp.Matrix(funcs)
    else:
        raise RuntimeError(f"f2m not support type of funcs: {type(funcs)}.")

def a2m(args: ArgArray) -> SympyMutableDenseMatrix:
    '''
    :param funcs: ArgArray, symbolic set constructed with values of `symbols`.

    :return: symbolic values updated with `sp.Matrix()`.
    '''
    if isinstance(args, (SympyMutableDenseMatrix, ArgType)):
        return sp.Matrix([args]) # prefer SympyMutableDenseMatrix to avoid enter into elif
    elif isinstance(args, (list, tuple)) and all(list(map(lambda x: isinstance(x, ArgType), args))):
        return sp.Matrix(args)
    else:
        raise RuntimeError(f"a2m not support type of args: {type(args)}")

def p2t(x_0: PointArray) -> PointArray:
    '''
    :param funcs: PointArray, numerical set constructed with numerical values in order of `args`.

    :return: numerical values set updated with `sp.Matrix()`.
    '''
    if isinstance(x_0, (float, int)):
        return (x_0,)
    elif isinstance(x_0, (list, tuple)) and all(list(map(lambda x: isinstance(x, (float, int)), x_0))):
        return x_0
    else:
        raise RuntimeError(f"p2t not support type of x_0: {type(x_0)}")

def h2h(hessian: NDArray) -> NDArray:
    '''
    :param hessian: NDArray, hessian matrix with format at `numpy.ndarray`.

    :return: returns a reversible hessian matrix.
    '''
    l = hessian.shape[0] # hessian.shape = (l, l)
    while 1:
        rank = np.linalg.matrix_rank(hessian)
        if rank == l:
            break
        else:
            hessian += np.identity(l)
    return hessian