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
from .._utils import plot_iteration
from .._convert import f2m, a2m, p2t, h2h

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

__all__ = ["steihaug_CG"]

def steihaug_CG(funcs: FuncArray, 
                args: ArgArray, 
                x_0: PointArray,
                verbose: bool=False, 
                draw: bool=True, 
                output_f: bool=False,  
                r0: float=1., 
                rmax: float=2., 
                eta: float=0.2, 
                p1: float=0.4, 
                p2: float=0.6, 
                gamma1: float=0.5, 
                gamma2: float=1.5,
                epsk: float=1e-6,
                epsilon: float=1e-6, 
                k: int=0) -> OutputType:
    '''
    :param funcr: FuncArray, current objective equation group constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param r0: float, the initial radius of gradient search used to update iteration points. default: float=1.0.
    :param rmax: float, the maximal radius of gradient search used to update iteration points. default: float=2.0.
    :param eta: float, threshold constraint required for controlling iteration point updates. default: float=0.2.
    :param p1: float, threshold for controlling whether r0 is updated by gamma1. default: float=0.4.
    :param p2: float, threshold for controlling whether r0 is updated by gamma2. default: float=0.6.
    :param gamma1: float, constant used for updating the value of r0 in the first if condition. default: float=0.5.
    :param gamma2: float, constant used for updating the value of r0 in the second if condition. default: float=1.5.
    :param epsk: float, the break epsilon of conjugate in searching for gradient. default: float=1e-6.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-6.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert eta >= 0
    assert r0 < rmax
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    from .._drive import steihaug
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    s0, f = [0 for _ in range(args.shape[0])], []
    while 1:
        reps = dict(zip(args, x_0))
        funv = np.array(funcs.subs(reps)).astype(DataType)
        f.append(funv[0][0])
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        grad = np.array(res.subs(reps)).astype(DataType)
        hessi = np.array(hes.subs(reps)).astype(DataType)
        hessi = h2h(hessi)
        dk = steihaug(s0, grad, -grad, hessi, r0, epsk).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            funvk = np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(DataType)
            pk = ((funv - funvk) / -(grad.dot(dk.T) + 0.5*((dk.dot(hessi)).dot(dk.T))))[0][0]
            if pk < p1:
                r0 *= gamma1
            else:
                if (pk > p2) or (np.linalg.norm(dk) == r0):
                    r0 = min(gamma2 * r0, rmax)
            if pk > eta:
                x_0 += dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "trust_region_steihaug_CG")
    return (x_0, k, f) if output_f is True else (x_0, k)