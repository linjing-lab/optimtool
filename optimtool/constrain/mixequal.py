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

__all__ = ["penalty_quadraticm", "penalty_L1", "lagrange_augmentedm"]

def penalty_quadraticm(funcs: FuncArray, 
                       args: ArgArray, 
                       cons_equal: FuncArray, 
                       cons_unequal: FuncArray, 
                       x_0: PointArray,
                       verbose: bool=False, 
                       draw: bool=True, 
                       output_f: bool=False, 
                       method: str="newton", 
                       sigma: float=10, 
                       p: float=0.6,
                       epsk: float=1e-6,
                       epsilon: float=1e-10, 
                       k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param cons_equal: FuncArray, all sets of equation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param cons_unequal: FuncArray, all sets of unequation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, unconstrained kernel used to drive the operation of finding the point of intermediate function. default: str='newton'.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=10.
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=0.6.
    :param epsk: float, used to set the precision to accelerate the completion of kernel. default: float=1e-6.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 0
    from .._kernel import kernel
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search, point, f = kernel(method), [], []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        reps = dict(zip(args, x_0))
        consv = np.array(cons_unequal.subs(reps)).astype(DataType)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons_unequal.T * consv + (sigma / 2) * cons_equal.T * cons_equal])
        x_0, _ = search(pe, args, tuple(x_0), draw=False, epsilon=epsk)
        k += 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
        sigma *= p
    plot_iteration(f, draw, "penalty_quadratic_mixequal")
    return (x_0, k, f) if output_f is True else (x_0, k)

def penalty_L1(funcs: FuncArray, 
               args: ArgArray, 
               cons_equal: FuncArray, 
               cons_unequal: FuncArray, 
               x_0: PointArray,
               verbose: bool=False, 
               draw: bool=True, 
               output_f: bool=False, 
               method: str="newton", 
               sigma: float=1, 
               p: float=0.6, 
               epsk: float=1e-6,
               epsilon: float=1e-10, 
               k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param cons_equal: FuncArray, all sets of equation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param cons_unequal: FuncArray, all sets of unequation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, unconstrained kernel used to drive the operation of finding the point of intermediate function. default: str='newton'.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=1.
    :param p: float, parameter to adjust the degree value of convergence named `sigma`. default: float=0.6.
    :param epsk: float, used to set the precision to accelerate the completion of kernel. default: float=1e-6.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-10.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 0
    from .._kernel import kernel
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search, point, f = kernel(method), [], []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        reps = dict(zip(args, x_0))
        consv_unequal = np.array(cons_unequal.subs(reps)).astype(DataType)
        consv_unequal = np.where(consv_unequal <= 0, consv_unequal, 1)
        consv_unequal = np.where(consv_unequal > 0, consv_unequal, 0)
        consv_equal = np.array(cons_equal.subs(reps)).astype(DataType)
        consv_equal = np.where(consv_equal <= 0, consv_equal, 1)
        consv_equal = np.where(consv_equal > 0, consv_equal, -1)
        pe = sp.Matrix([funcs + sigma * cons_unequal.T * consv_unequal + sigma * cons_equal.T * consv_equal])
        x_0, _ = search(pe, args, tuple(x_0), draw=False, epsilon=epsk)
        sigma, k = p * sigma, k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "penalty_L1")
    return (x_0, k, f) if output_f is True else (x_0, k)

def lagrange_augmentedm(funcs: FuncArray, 
                        args: ArgArray, 
                        cons_equal: FuncArray, 
                        cons_unequal: FuncArray, 
                        x_0: PointArray,
                        verbose: bool=False, 
                        draw: bool=True, 
                        output_f: bool=False, 
                        method: str="newton", 
                        lamk: float=6, 
                        muk: float=10, 
                        sigma: float=8, 
                        alpha: float=0.5, 
                        beta: float=0.7, 
                        p: float=2, 
                        etak: float=1e-3, 
                        epsilon: float=1e-4, 
                        k: int=0) -> OutputType:
    '''
    :param funcs: FuncArray, current objective equation constructed with values of `symbols` according to rules.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param cons_equal: FuncArray, all sets of equation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param cons_unequal: FuncArray, all sets of unequation constraints without order in a `list`, `tuple` or single `FuncArray`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param method: str, unconstrained kernel used to drive the operation of finding the point of intermediate function. default: str='newton'.
    :param lamk: float, constant used to adjust the weight of equation constraints. default: float=6.
    :param muk: float, controlled parameter with unequality constrained sigma values. default: float=10.
    :param sigma: float, penalty factor used to set the degree of convergence of `funcs`. default: float=8.
    :param alpha: float, value to adjust epsilonk combined with sigma value. default: float=0.5.
    :param beta: float, value used in continue execution to adjust epsilonk. default: float=0.7.
    :param p: float, value to adjust the degree value of convergence named `sigma`. default: float=2.
    :param etak: float, used to set the precision to measure the gradient of funcs. default: float=1e-3.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-4.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert sigma > 0
    assert p > 1
    assert alpha > 0 
    assert alpha <= beta
    assert beta < 1
    from .._kernel import kernel
    from .._drive import cons_unequal_L, v_k, renew_mu_k
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search, f = kernel(method), []
    lamk = np.array([lamk for _ in range(cons_equal.shape[0])]).reshape(cons_equal.shape[0], 1)
    muk = np.array([muk for _ in range(cons_unequal.shape[0])]).reshape(cons_unequal.shape[0], 1)
    while 1:
        f.append(get_value(funcs, args, x_0))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = cons_unequal_L(cons_unequal, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * (cons_equal.T * cons_equal + cons_uneuqal_modifyed) + cons_equal.T * lamk])
        x_0, _ = search(L, args, tuple(x_0), draw=False, epsilon=epsilonk)
        k += 1
        vkx = v_k(cons_equal, cons_unequal, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)) <= etak):
                f.append(get_value(funcs, args, x_0))
                if verbose:
                    print("{}\t{}\t{}".format(x_0, f[-1], k))
                break
            else:
                lamk += sigma * np.array(cons_equal.subs(dict(zip(args, x_0)))).astype(DataType)
                muk = renew_mu_k(cons_unequal, args, muk, sigma, x_0)
                etak /= sigma
                epsilonk /= sigma**beta
        else:
            sigma *= p
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    plot_iteration(f, draw, "lagrange_augmented_mixequal")
    return (x_0, k, f) if output_f is True else (x_0, k)