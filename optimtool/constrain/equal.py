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

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

__all__ = ["penalty_quadratice", "lagrange_augmentede"]

def penalty_quadratice(funcs: FuncArray, 
                       args: ArgArray, 
                       cons: FuncArray, 
                       x_0: PointArray,
                       verbose: bool=False, 
                       draw: bool=True, 
                       output_f: bool=False, 
                       method: str="newton", 
                       sigma: float=10., 
                       p: float=2., 
                       epsk: float=1e-4, 
                       epsilon: float=1e-6, 
                       k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param cons: FuncArray, all sets of equation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, unconstrained kernel used to drive the operation of finding the point of intermediate function. default: str='newton'.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=10.0.
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=2.0.
    :param epsk: float, used to set the precision to accelerate the completion of kernel. default: float=1e-4.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-6.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 1
    from .._kernel import kernel
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, point, f = kernel(method), [], []
    sig = sp.symbols("sig")
    pen = funcs + (sig / 2) * cons.T * cons
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        pe = pen.subs(sig, sigma)
        x_0, _ = search(pe, args, tuple(x_0), draw=False, epsilon=epsk)
        sigma, k = p * sigma, k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "penalty_quadratic_equal")     
    return (x_0, k, f) if output_f is True else (x_0, k)

def lagrange_augmentede(funcs: FuncArray, 
                        args: ArgArray, 
                        cons: FuncArray, 
                        x_0: PointArray,
                        verbose: bool=False, 
                        draw: bool=True, 
                        output_f: bool=False, 
                        method: str="newton", 
                        lamk: float=6., 
                        sigma: float=10., 
                        p: float=2., 
                        etak: float=1e-4, 
                        epsilon: float=1e-6, 
                        k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param cons: FuncArray, all sets of equation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, unconstrained kernel used to drive the operation of finding the point of intermediate function. default: str='newton'.
    :param lamk: float, the initial value of the elements in the initial penalty vector. default: float=6.0.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=10.0.
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=2.0.
    :param epak: float, used to set the precision to accelerate the completion of kernel. default: float=1e-4.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-6.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 1
    from .._kernel import kernel
    search, f = kernel(method), []
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    lamk = np.array([lamk for _ in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while 1:
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        L = sp.Matrix([funcs + (sigma / 2) * cons.T * cons + cons.T * lamk])
        x_0, _ = search(L, args, tuple(x_0), draw=False, epsilon=etak)
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(DataType)
        lamk = lamk + sigma * consv
        sigma, k = p * sigma, k + 1
        if np.linalg.norm(consv) <= epsilon:
            f.append(get_value(funcs, args, x_0))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "lagrange_augmented_equal")
    return (x_0, k, f) if output_f is True else (x_0, k)