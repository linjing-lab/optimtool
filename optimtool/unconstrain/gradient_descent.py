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
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t

from .._typing import FuncArray, ArgArray, PointArray, DataType, OutputType

__all__ = ["solve", "steepest", "barzilar_borwein"]

def solve(funcs: FuncArray, 
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
    m = sp.symbols("m")
    arg, f = sp.Matrix([m]), []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        dk = -np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            xt = x_0 + m * dk[0]
            h = funcs.subs(dict(zip(args, xt))).jacobian(arg)
            mt = sp.solve(h)
            x_0 += (mt[m] * dk[0]).astype(DataType)
            k += 1
        else:
            break
    plot_iteration(f, draw, "gradient_descent_solve")
    return (x_0, k, f) if output_f is True else (x_0, k)

def steepest(funcs: FuncArray, 
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
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    from .._kernel import linear_search
    search, f = linear_search(method), []
    res = funcs.jacobian(args) # gradient
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        dk = -np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "gradient_descent_steepest")
    return (x_0, k, f) if output_f is True else (x_0, k)

def barzilar_borwein(funcs: FuncArray, 
                     args: ArgArray, 
                     x_0: PointArray,
                     verbose: bool=False, 
                     draw: bool=True, 
                     output_f: bool=False, 
                     method: str="Grippo", 
                     c1: float=0.6, 
                     beta: float=0.6,
                     M: int=20,
                     eta: float=0.6,
                     alpha: float=1, 
                     epsilon: float=1e-10, 
                     k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, nonmonotone line search kernel used to drive the operation of finding best alpha default: str='Grippo'.
    :param c1: float, constant used to constrain alpha adjusted frequency with interval at (0, 1). default: float=0.6.
    :param beta: float, factor used to expand alpha for adapting to alphas interval. default: float=0.6
    :param alpha: float, initial step size for nonmonotonic line search method with assert `> 0`. default: float=1.
    :param M: int, constant used to control the inner `max` process of `Grippo`. default: int=20.
    :param eta: float, constant used to control `C_k` process of `ZhangHanger`. default: float=0.6.
    :param alpha: float=1, the initial step size of the main algorithm. default: float=1.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    assert M > 0
    assert eta > 0
    assert eta < 1
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    from .._kernel import nonmonotonic_search
    search, constant = nonmonotonic_search(method, M, eta)
    res, point, f = funcs.jacobian(args), [], []
    while 1:
        point.append(x_0)
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        dk = -np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk, k, point, c1, beta, alpha, constant)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) + dk
            alpha_up = delta.dot(delta.T)
            alpha_down = delta.dot(yk.T)
            if alpha_down != 0:
                alpha = alpha_up / alpha_down
            k += 1
        else:
            break
    plot_iteration(f, draw, "gradient_descent_barzilar_borwein_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)