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

from ..base import np
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t, h2h

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

__all__ = ["bfgs", "dfp", "L_BFGS"]

def bfgs(funcs: FuncArray, 
         args: ArgArray, 
         x_0: PointArray,
         verbose: bool=False, 
         draw: bool=True, 
         output_f: bool=False, 
         method: str="wolfe",  
         epsilon: float=1e-10, 
         k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, linear search kernel used to drive the operation of finding best aplha. default: str='wolfe'.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hessian = np.array(hes.subs(dict(zip(args, x_0)))).astype(DataType)
    hessian = h2h(hessian)
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        gradient = np.array(res.subs(reps)).astype(DataType)
        dk = -np.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 += delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) - np.array(res.subs(reps)).astype(DataType)
            if yk.all != 0:
                hessian += (yk.T).dot(yk) / sk.dot(yk.T) - (hessian.dot(sk.T)).dot((hessian.dot(sk.T)).T) / sk.dot((hessian.dot(sk.T)))
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_quasi_bfgs_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

def dfp(funcs: FuncArray, 
        args: ArgArray, 
        x_0: PointArray,
        verbose: bool=False, 
        draw: bool=True, 
        output_f: bool=False, 
        method: str="wolfe",  
        epsilon: float=1e-10, 
        k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, linear search kernel used to drive the operation of finding best aplha. default: str='wolfe'.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hessian = np.array(hes.subs(dict(zip(args, x_0)))).astype(DataType)
    hessian = h2h(hessian)
    hessiani = np.linalg.inv(hessian)
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        gradient = np.array(res.subs(reps)).astype(DataType)
        dk = -hessiani.dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 += delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) - np.array(res.subs(reps)).astype(DataType)
            if yk.all != 0:
                hessiani = hessiani - (hessiani.dot(yk.T)).dot((hessiani.dot(yk.T)).T) / yk.dot(hessiani.dot(yk.T)) + (sk.T).dot(sk) / yk.dot(sk.T)
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_quasi_dfp_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

def L_BFGS(funcs: FuncArray, 
           args: ArgArray, 
           x_0: PointArray,
           verbose: bool=False, 
           draw: bool=True, 
           output_f: bool=False, 
           method: str="wolfe", 
           m: int=6, 
           epsilon: float=1e-10, 
           k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, linear search kernel used to drive the operation of finding best aplha. default: str='wolfe'.
    :param m: int, threshold for controlling the cursor range of double_loop iteration process. default: int=6.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert m > 0
    from .._drive import double_loop
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search = linear_search(method)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    l = hes.shape[0]
    f, s, y, p = [], [], [], []
    gamma = []
    gamma.append(1)
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        Hkm = gamma[k] * np.identity(l)
        grad = np.array(res.subs(reps)).astype(DataType)
        dk = -double_loop(grad, p, s, y, m, k, Hkm).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alphak = search(funcs, args, x_0, dk)
            x_0 = x_0 + alphak * dk[0]
            if k > m:
                s[k-m] = np.empty((1, l))
                y[k-m] = np.empty((1, l))
            sk = alphak * dk
            s.append(sk)
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) - grad
            y.append(yk)
            pk = 1 / yk.dot(sk.T)
            p.append(pk)
            gammak = sk.dot(sk.T) / yk.dot(yk.T)
            gamma.append(gammak)
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "newton_quasi_L_BFGS_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)