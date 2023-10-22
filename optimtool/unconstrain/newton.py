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

__all__ = ["classic", "modified", "CG"]

def classic(funcs: FuncArray, 
            args: ArgArray, 
            x_0: PointArray,
            verbose: bool=False, 
            draw: bool=True, 
            output_f: bool=False, 
            epsilon: float=1e-10, 
            k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    res = funcs.jacobian(args) # gradient
    hes, f = res.jacobian(args), []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hessian = np.array(hes.subs(reps)).astype(DataType)
        dk = -np.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            x_0 += dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_classic")        
    return (x_0, k, f) if output_f is True else (x_0, k)

def modified(funcs: FuncArray, 
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
    res = funcs.jacobian(args) # graident
    hes = res.jacobian(args) # hessian
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hessian = np.array(hes.subs(reps)).astype(DataType) # hessian: from `object` to `float`
        hessian = h2h(hessian)
        dk = -np.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_modified_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

def CG(funcs: FuncArray, 
       args: ArgArray, 
       x_0: PointArray,
       verbose: bool=False, 
       draw: bool=True, 
       output_f: bool=False, 
       method: str="wolfe",
       eps: float=1e-3, 
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
    :param eps: float, the precision set for solve conjugate gradient method to obtain the next dk. default: float=1e-3.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    from .._drive import conjugate
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res = funcs.jacobian(args) # gradient
    hes, dk0 = res.jacobian(args), np.zeros((args.shape[0], 1)) # hessian and initial dk
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hessian = np.array(hes.subs(reps)).astype(DataType)
        dk = conjugate(hessian, -gradient, dk0, eps).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_CG_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)