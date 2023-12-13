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

__all__ = ["penalty_quadraticu", "lagrange_augmentedu"]

def penalty_quadraticu(funcs: FuncArray, 
                       args: ArgArray, 
                       cons: FuncArray, 
                       x_0: PointArray,
                       verbose: bool=False, 
                       draw: bool=True, 
                       output_f: bool=False, 
                       method: str="newton", 
                       sigma: float=10., 
                       p: float=0.4, 
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
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=0.4.
    :param epsk: float, used to set the precision to accelerate the completion of kernel. default: float=1e-4.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-6.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 0
    assert p < 1
    from .._kernel import kernel
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, point, f = kernel(method), [], []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(DataType)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons.T * consv])
        x_0, _ = search(pe, args, tuple(x_0), draw=False, epsilon=epsk)
        k += 1
        sigma *= p
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "penalty_quadratic_unequal") 
    return (x_0, k, f) if output_f is True else (x_0, k)

def lagrange_augmentedu(funcs: FuncArray, 
                        args: ArgArray, 
                        cons: FuncArray, 
                        x_0: PointArray,
                        verbose: bool=False, 
                        draw: bool=True, 
                        output_f: bool=False, 
                        method: str="newton", 
                        muk: float=10., 
                        sigma: float=8., 
                        alpha: float=0.2, 
                        beta: float=0.7, 
                        p: float=2., 
                        eta: float=1e-1, 
                        epsilon: float=1e-4, 
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
    :param muk: float, controlled parameter with unequality constrained sigma values. default: float=10.0.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=8.0.
    :param alpha: float, value to adjust epsilonk combined with sigma value. default: float=0.2.
    :param beta: float, value used in continue execution to adjust epsilonk. default: float=0.7.
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=2.0.
    :param eta: float, used to set the precision to measure the gradient of funcs. default: float=1e-1.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-4.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 1
    assert alpha > 0 
    assert alpha <= beta
    assert beta > 0
    assert beta < 1
    from .._kernel import kernel
    from .._drive import cons_unequal_L, renew_mu_k, v_k
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, f = kernel(method), []
    muk = np.array([muk for _ in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while 1:
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = cons_unequal_L(cons, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * cons_uneuqal_modifyed])
        x_0, _ = search(L, args, tuple(x_0), draw=False, epsilon=epsilonk)
        k += 1
        vkx = v_k(None, cons, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)) <= eta):
                f.append(get_value(funcs, args, x_0))
                if verbose:
                    print("{}\t{}\t{}".format(x_0, f[-1], k))
                break
            else:
                muk = renew_mu_k(cons, args, muk, sigma, x_0)
                etak /= sigma
                epsilonk /= sigma**beta
        else:
            sigma *= p
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    plot_iteration(f, draw, "lagrange_augmented_unequal")
    return (x_0, k, f) if output_f is True else (x_0, k)