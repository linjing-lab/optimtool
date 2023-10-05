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
from .._convert import f2m, a2m, p2t

from .._typing import FuncArray, ArgArray, PointArray, DataType, OutputType

__all__ = ["normal", "variant"]

def normal(funcs: FuncArray,
		   args: ArgArray, 
		   x_0: PointArray,
		   mu: float=1e-3, 
		   proxim: str="L1",
		   tk: float=0.02,  
		   verbose: bool=False, 
		   draw: bool=True,
		   output_f: bool=False,
		   epsilon: float=1e-10,
		   k: int=0) -> OutputType:
	'''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param mu: float, regularization parameter acting on proximity operator selected from proxim. default: float=1e-3.
    :param proxim: str, proximity operator set by _proxim.py and set_proxim from .._kernel. default: str="L1".
    :param tk: float, fixed step which need to be lower conform to lipschitz continuity condition. default: float=0.02.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert tk > 0 and tk < 1
    assert mu > 0 and mu < 1
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    from .._kernel import set_proxim
    proximo, point, f = set_proxim(proxim), [], []
    res = funcs.jacobian(args) # gradient
    while 1:
    	point.append(np.array(x_0))
    	f.append(get_value(funcs, args, x_0, mu, proxim))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
    	dk = -np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
        	yk = x_0 if k == 0 else x_0 + (k - 1) * (x_0 - point[k-1]) / (k + 2)
        	dky = -np.array(res.subs(dict(zip(args, yk)))).astype(DataType)
        	delta = yk + tk * dky[0]
         	x_0 = proximo(delta, mu, tk)
         	k += 1
         else:
         	break
    plot_iteration(f, draw, "FISTA_normal")
    return (x_0, k, f) if output_f is True else (x_0, k)

def variant(funcs: FuncArray,
		    args: ArgArray, 
		    x_0: PointArray,
		    mu: float=1e-3, 
		    proxim: str="L1",
		    tk: float=0.02,  
		    verbose: bool=False, 
		    draw: bool=True,
		    output_f: bool=False,
		    epsilon: float=1e-10,
		    k: int=0) -> OutputType:
	'''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param mu: float, regularization parameter acting on proximity operator selected from proxim. default: float=1e-3.
    :param proxim: str, proximity operator set by _proxim.py and set_proxim from .._kernel. default: str="L1".
    :param tk: float, fixed step which need to be lower conform to lipschitz continuity condition. default: float=0.02.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert tk > 0 and tk < 1
    assert mu > 0 and mu < 1
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    from .._kernel import set_proxim
    from .._drive import gammak
    proximo, point, f = set_proxim(proxim), [], []
    res = funcs.jacobian(args) # gradient
    vkx = lambda k: x_0 if k == 0 else point[k-1] + (x_0 - point[k-1]) / gk
    while 1:
    	point.append(np.array(x_0))
    	f.append(get_value(funcs, args, x_0, mu, proxim))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
    	dk = -np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
        	gk = gammak(k)
        	yk = (1 - gk) * x_0 + gk * vkx(k)
        	dky = -np.array(res.subs(dict(zip(args, yk)))).astype(DataType)
        	delta = yk + tk * dky[0]
         	x_0 = proximo(delta, mu, tk)
         	k += 1
         else:
         	break
    plot_iteration(f, draw, "FISTA_variant")
    return (x_0, k, f) if output_f is True else (x_0, k)