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
from .._convert import a2m, p2t
from .._utils import get_value, plot_iteration

from .._typing import NDArray, ArgArray, PointArray, OutputType, DataType

__all__ = ["gradient", "subgradient", "approximate_point"]

def gradient(A: NDArray, 
             b: NDArray, 
             mu: float, 
             args: ArgArray, 
             x_0: PointArray,
             verbose: bool=False, 
             draw: bool=True, 
             output_f: bool=False, 
             delta: float=10., 
             alp: float=1e-3, 
             epsilon: float=1e-2, 
             k: int=0) -> OutputType:
    '''
    :param A: NDArray, matrix A with size m*n acting on x.
    :param b: NDArray, A constant vector b of size m*1 acting on Ax.
    :param mu: float, the regularization constant mu acting on |x|.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param delta: float, value used to adjust the constant influence of mu. default: float=10.0.
    :param alp: float, initial update step size acting on smooth gradient. default: float=1e-3.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-2
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert delta > 0
    assert alp > 0
    from .._drive import get_f_delta_gradient
    args, x_0 = a2m(args), p2t(x_0)
    assert args.shape[0] == len(x_0)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    L = np.linalg.norm((A.T).dot(A)) + mu / delta
    assert alp <= 1 / L
    point, f = [], []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0, mu, "L1"))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        resv = np.array(res.subs(reps)).astype(DataType)
        argsv = np.array(args.subs(reps)).astype(DataType)
        g = get_f_delta_gradient(resv, argsv, mu, delta)
        x_0 = x_0 - alp * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0, mu))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "Lasso_gradient_descent")
    return (x_0, k, f) if output_f is True else (x_0, k)

def subgradient(A: NDArray, 
                b: NDArray, 
                mu: float, 
                args: ArgArray, 
                x_0: PointArray,
                verbose: bool=False, 
                draw: bool=True, 
                output_f: bool=False, 
                alphak: float=2e-2, 
                epsilon: float=1e-3, 
                k: int=0) -> OutputType:
    '''
    :param A: NDArray, matrix A with size m*n acting on x.
    :param b: NDArray, A constant vector b of size m*1 acting on Ax.
    :param mu: float, The regularization constant mu acting on |x|.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param alphak: float, limit the initial constant for the next update of iteration point. default: float=2e-2. 
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-3.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    assert alphak > 0
    from .._drive import get_subgradient
    args, x_0 = a2m(args), p2t(x_0)
    assert args.shape[0] == len(x_0)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    point, f = [], []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0, mu, "L1"))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        resv = np.array(res.subs(reps)).astype(DataType)
        argsv = np.array(args.subs(reps)).astype(DataType)
        g = get_subgradient(resv, argsv, mu)
        alpha = alphak / np.sqrt(k + 1)
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0, mu))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "Lasso_subgradient")
    return (x_0, k, f) if output_f is True else (x_0, k)

def approximate_point(A: NDArray, 
                      b: NDArray, 
                      mu: float, 
                      args: ArgArray, 
                      x_0: PointArray,
                      verbose: bool=False, 
                      draw: bool=True, 
                      output_f: bool=False, 
                      epsilon: float=1e-4, 
                      k: int=0) -> OutputType:
    '''
    :param A: NDArray, matrix A with size m*n acting on x.
    :param b: NDArray, constant vector b with size m* 1 acting on Ax.
    :param mu: float, The regularization constant mu acting on |x|.
    :param args: ArgArray, symbol parameters composed with values of `symbols` in a `list` or `tuple`.
    :param x_0: PointArray, numerical iteration point in a `list` or `tuple` according to the order of values in `args`.
    :param verbose: bool, iteration point, function value, numbers of iteration after the k-th iteration. default: bool=False.
    :param draw: bool, use `bool` to control whether to draw visual images. default: bool=True.
    :param output_f: bool, use `bool` to control whether to obtain iterative values of `funcs`. default: bool=False.
    :param epsilon: float, used to set the precision of stopping the overall algorithm. default: float=1e-4.
    :param k: int, iterative times is used to measure the difficulty of learning the `funcs` in the algorithm. default: int=0.

    :return: final convergenced point and iterative times, (iterative values in a list).
    '''
    args, x_0 = a2m(args), p2t(x_0)
    assert args.shape[0] == len(x_0)
    values, _ = np.linalg.eig((A.T).dot(A))
    lambda_ma = max(values)
    tk = 1 / np.real(lambda_ma) if isinstance(lambda_ma, complex) else 1 / lambda_ma
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    point, f = [], []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(x_0)
        f.append(get_value(funcs, args, x_0, mu, "L1"))
        if verbose:
            print("{}\t{}\t{}".format(x_0, f[-1], k))
        grad = np.array(res.subs(reps)).astype(DataType)
        x_0 = np.sign(x_0 - tk * grad[0]) * [max(i, 0) for i in np.abs(x_0 - tk * grad[0]) - tk * mu]
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            f.append(get_value(funcs, args, x_0, mu))
            if verbose:
                print("{}\t{}\t{}".format(x_0, f[-1], k))
            break
    plot_iteration(f, draw, "Lasso_approximate_point_gradient")
    return (x_0, k, f) if output_f is True else (x_0, k)