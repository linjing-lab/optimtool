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

from .base import np
from ._typing import List, NDArray, SympyMutableDenseMatrix, DataType, IterPointType

__all__ = ["armijo", "goldstein", "wolfe", "Grippo", "ZhangHanger"]

def armijo(funcs: SympyMutableDenseMatrix, 
           args: SympyMutableDenseMatrix, 
           x_0: IterPointType, 
           d: NDArray, 
           gamma: float=0.5, 
           c: float=0.1) -> float:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function with `convert` process used for search alpha.
    :param args: SympyMutableDenseMatrix, symbolic set with order in a list to construct `dict(zip(args, x_0))`.
    :param x_0: IterPointType, numerical values in a 'list` or 'tuple` according to the order of `args`.
    :param d: NDArray, current gradient descent direction with format at `numpy.ndarray`.
    :param gamma: float, factor used to adjust alpha with interval at (0, 1). default: float=0.5.
    :param c: float, constant used to constrain alpha adjusted frequency with interval at (0, 1). default: float=0.1.

    :return: best alpha with format at `float`.
    '''
    assert gamma > 0
    assert gamma < 1
    assert c > 0
    assert c < 1
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            break
        else:
            alpha *= gamma
    return alpha

def goldstein(funcs: SympyMutableDenseMatrix, 
              args: SympyMutableDenseMatrix, 
              x_0: IterPointType, 
              d: NDArray, 
              c: float=0.2, 
              alphas: float=0., 
              alphae: float=10., 
              t: float=1.2, 
              eps: float=1e-3) -> float:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function with `convert` process used for search alpha.
    :param args: SympyMutableDenseMatrix, symbolic set with order in a list to construct `dict(zip(args, x_0))`.
    :param x_0: IterPointType, numerical values in a 'list` or 'tuple` according to the order of `args`.
    :param d: NDArray, current gradient descent direction with format at `numpy.ndarray`.
    :param c: float, constant used to constrain alpha adjusted frequency with interval at (0, 0.5). default: float=0.2.
    :param alphas: float, left search endpoint for alpha with value assert as `> 0`. default: float=0.
    :param alphae: float, right search endpoint for alpha with value assert as `> alphas`. default: float=10.
    :param t: float, factor used to expand alpha for adapting to alphas interval. default: float=1.2.
    :param eps: float, the precision set to stopp search alpha in (alphas, alphae). default: float=1e-3.

    :return: best alpha with format at `float`.
    '''
    assert c > 0
    assert c < 0.5
    assert alphas >= 0
    assert alphas < alphae
    assert t > 0
    assert eps > 0
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            if f1 >= f0 + (1 - c)*alpha*res0.dot(d.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
                if alphae < np.inf:
                    alpha = 0.5 * (alphas + alphae)
                else:
                    alpha *= t
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas - alphae) < eps:
            break
    return alpha

def wolfe(funcs: SympyMutableDenseMatrix, 
          args: SympyMutableDenseMatrix, 
          x_0: IterPointType, 
          d: NDArray, 
          c1: float=0.3, 
          c2: float=0.5, 
          alphas: float=0., 
          alphae: float=2., 
          eps: float=1e-3) -> float:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function with `convert` process used for search alpha.
    :param args: SympyMutableDenseMatrix, symbolic set with order in a list to construct `dict(zip(args, x_0))`.
    :param x_0: IterPointType, numerical values in a 'list` or 'tuple` according to the order of `args`.
    :param d: NDArray, current gradient descent direction with format at `numpy.ndarray`.
    :param c1: float, first constant used to constrain alpha adjusted frequency with interval at (0, 1). default: float=0.3.
    :param c2: float, second constant used to constrain alpha adjusted frequency with interval at (0, 1). default: float=0.5.
    :param alphas: float, left search endpoint for alpha with value assert as `> 0`. default: float=0.
    :param alphae: float, right search endpoint for alpha with value assert as `> alphas`. default: float=2.
    :param eps: float, the precision set to stopp search alpha in (alphas, alphae). default: float=1e-3.

    :return: best alpha with format at `float`.
    '''
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas >= 0
    assert alphas < alphae
    assert eps > 0
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c1*alpha*res0.dot(d.T):
            res1 = np.array(res.subs(dict(zip(args, x)))).astype(DataType)
            if res1.dot(d.T) >= c2*res0.dot(d.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas - alphae) < eps:
            break
    return alpha

# coordinate with `barzilar_borwein`.
def Grippo(funcs: SympyMutableDenseMatrix, 
           args: SympyMutableDenseMatrix, 
           x_0: IterPointType, 
           d: NDArray, 
           k: int, 
           point: List[IterPointType], 
           c1: float, 
           beta: float, 
           alpha: float, 
           M: int) -> float:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function with `convert` process used for search alpha.
    :param args: SympyMutableDenseMatrix, symbolic set with order in a list to construct `dict(zip(args, x_0))`.
    :param x_0: IterPointType, numerical values in a 'list` or 'tuple` according to the order of `args`.
    :param d: NDArray, current gradient descent direction with format at `numpy.ndarray`.
    :param k: int, current number of iterative process in `barzilar_borwein` method.
    :param c1: float, constant used to constrain alpha adjusted frequency with interval at (0, 1).
    :param beta: float, factor used to expand alpha for adapting to alphas interval.
    :param alpha: float, initial step size for nonmonotonic line search method with assert `> 0`.
    :param M: int, constant to control the inner `max` proccess with assert `>= 0`.

    :return: best alpha with format at `float`.
    '''
    assert M > 0
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        fk = -np.inf
        for j in range(min(k, M) + 1):
            fk = max(fk, np.array(funcs.subs(dict(zip(args, point[k-j])))).astype(DataType))
        if f1 <= fk + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha *= beta
    return alpha

def ZhangHanger(funcs: SympyMutableDenseMatrix, 
                args: SympyMutableDenseMatrix, 
                x_0: IterPointType, 
                d: NDArray, 
                k: int, 
                point: List[IterPointType], 
                c1: float, 
                beta: float, 
                alpha: float, 
                eta: float) -> float:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function with `convert` process used for search alpha.
    :param args: SympyMutableDenseMatrix, symbolic set with order in a list to construct `dict(zip(args, x_0))`.
    :param x_0: IterPointType, numerical values in a 'list` or 'tuple` according to the order of `args`.
    :param d: NDArray, current gradient descent direction with format at `numpy.ndarray`.
    :param k: int, current number of iterative process in `barzilar_borwein` method.
    :param c1: float, constant used to constrain alpha adjusted frequency with interval at (0, 1).
    :param beta: float, factor used to expand alpha for adapting to alphas interval.
    :param alpha: float, initial step size for nonmonotonic line search method with assert `> 0`.
    :param eta: float, constant used to control `C_k` process with interval at (0, 1).

    :return: best alpha with format at `float`.
    '''
    assert alpha > 0
    assert eta > 0
    assert eta < 1
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    from ._drive import C_k
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        Ck = C_k(funcs, args, point, eta, k)
        if f1 <= Ck + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha *= beta
    return alpha