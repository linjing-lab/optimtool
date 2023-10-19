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

from ..base import np, sp
from .._convert import f2m, a2m, p2t, h2h
from .._utils import get_value, plot_iteration

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

__all__ = ["gauss_newton", "levenberg_marquardt"]

def gauss_newton(funcr: FuncArray, 
                 args: ArgArray, 
                 x_0: PointArray,
                 verbose: bool=False, 
                 draw: bool=True, 
                 output_f: bool=False, 
                 method: str="wolfe", 
                 epsilon: float=1e-10, 
                 k: int=0) -> OutputType:
    '''
    :param funcr: FuncArray, current objective equation group constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcr`. default: bool=False.
    :param method: str, linear search kernel used to drive the operation of finding best aplha. default: str='wolfe'.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcr` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    from .._kernel import linear_search
    funcr, args, x_0 = f2m(funcr), a2m(args), p2t(x_0)
    assert funcr.shape[0] > 1 and funcr.shape[1] ==1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res, funcs = funcr.jacobian(args), sp.Matrix([(1/2)*funcr.T*funcr])
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(DataType)
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        jk = np.array(res.subs(reps)).astype(DataType)
        q, r = np.linalg.qr(jk)
        dk = np.linalg.inv(r).dot(-(q.T).dot(rk)).reshape(1,-1) # operate with x_0
        if np.linalg.norm(dk) > epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "nonlinear_least_square_gauss_newton_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

def levenberg_marquardt(funcr: FuncArray, 
                        args: ArgArray, 
                        x_0: PointArray,
                        verbose: bool=False, 
                        draw: bool=True, 
                        output_f: bool=False, 
                        lamk: float=1., 
                        eta: float=0.2, 
                        p1: float=0.4, 
                        p2: float=0.9, 
                        gamma1: float=0.7, 
                        gamma2: float=1.3,
                        epsk: float=1e-6, 
                        epsilon: float=1e-10, 
                        k: int=0) -> OutputType:
    '''
    :param funcr: FuncArray, current objective equation group constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcr`. default: bool=False.
    :param lamk: float, the initial factor acting on the product of first-order residual matrices. default: float=1.0.
    :param eta: float, threshold constraint required for controlling iteration point updates. default: float=0.2.
    :param p1: float, threshold for controlling whether lamk is updated by gamma2. default: float=0.4.
    :param p2: float, threshold for controlling whether lamk is updated by gamma1. default: float=0.9.
    :param gamma1: float, constant used for updating the value of lamk in the first if condition. default: float=0.7.
    :param gamma2: float, constant used for updating the value of lamk in the second if condition. default: float=1.3.
    :param epsk: float, the break epsilon of conjugate in searching for gradient. default: float=1e-6.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcr` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert eta >= 0
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    from .._drive import conjugate
    funcr, args, x_0 = f2m(funcr), a2m(args), p2t(x_0)
    assert funcr.shape[0] > 1 and funcr.shape[1] ==1 and args.shape[0] == len(x_0)
    res, funcs = funcr.jacobian(args), sp.Matrix([(1/2)*funcr.T*funcr])
    resf = funcs.jacobian(args)
    hess, dk0, f = resf.jacobian(args), np.zeros((args.shape[0], 1)), []
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(DataType)
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        jk = np.array(res.subs(reps)).astype(DataType)
        dk = conjugate((jk.T).dot(jk) + lamk, -((jk.T).dot(rk)).reshape(1, -1), dk0, epsk)
        pk_up = np.array(funcs.subs(reps)).astype(DataType) - np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(DataType)
        grad_f = np.array(resf.subs(reps)).astype(DataType)
        hess_f = np.array(hess.subs(reps)).astype(DataType)
        hess_f = h2h(hess_f)
        pk_down = -(grad_f.dot(dk.T) + 0.5*((dk.dot(hess_f)).dot(dk.T)))
        pk = pk_up / pk_down
        if np.linalg.norm(dk) >= epsilon:
            if pk < p1:
                lamk = gamma2 * lamk
            else:
                if pk > p2:
                    lamk = gamma1 * lamk
            if pk > eta:
                x_0 = x_0 + dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "nonlinear_least_square_levenberg_marquardt")        
    return (x_0, k, f) if output_f is True else (x_0, k)