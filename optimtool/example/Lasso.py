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

__all__ = ['gradient_descent', 'subgradient', 'penalty', 'approximate_point_gradient']

import numpy as np
import sympy as sp
from .._convert import a2m
from .._utils import get_value, plot_iteration

from .._typing import NDArray, ArgArray, PointArray, Optional, OutputType, DataType

def gradient_descent(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, delta: Optional[float]=10, alp: Optional[float]=1e-3, epsilon: Optional[float]=1e-2, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    A : NDArray
        m*n维数 参数矩阵
        
    b : NDArray
        m*1维数 参数矩阵
        
    mu : Optional[float]
        正则化参数
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    delta : Optional[float]
        常数
        
    alp : Optional[float]
        步长阈值
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._drive import get_f_delta_gradient
    args = a2m(args)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    L = np.linalg.norm((A.T).dot(A)) + mu / delta
    point = []
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0, mu))
        resv = np.array(res.subs(reps)).astype(DataType)
        argsv = np.array(args.subs(reps)).astype(DataType)
        g = get_f_delta_gradient(resv, argsv, mu, delta)
        alpha = alp
        assert alpha <= 1 / L
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0, mu))
            break
    plot_iteration(f, draw, "Lasso_gradient_decent")
    return (x_0, k, f) if output_f is True else (x_0, k)

'''
次梯度算法
'''
def subgradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, alphak: Optional[float]=2e-2, epsilon: Optional[float]=1e-3, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    A : NDArray
        m*n维数 参数矩阵
        
    b : NDArray
        m*1维数 参数矩阵
        
    mu : float
        正则化参数
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    alphak : Optional[float]
        自适应步长参数
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._drive import get_subgradient
    args = a2m(args)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    point = []
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0, mu))
        resv = np.array(res.subs(reps)).astype(DataType)
        argsv = np.array(args.subs(reps)).astype(DataType)
        g = get_subgradient(resv, argsv, mu)
        alpha = alphak / np.sqrt(k + 1)
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0, mu))
            break
    plot_iteration(f, draw, "Lasso_subgradient")
    return (x_0, k, f) if output_f is True else (x_0, k)

'''
罚函数法
'''
def penalty(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, gamma: Optional[float]=0.01, epsilon: Optional[float]=1e-6, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    A : NDArray
        m*n维数 参数矩阵
        
    b : NDArray
        m*1维数 参数矩阵
        
    mu : float
        正则化参数
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    gamma : Optional[float]
        因子
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert gamma < 1
    assert gamma > 0
    args = a2m(args)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    f = []
    while mu >= epsilon:
        f.append(get_value(funcs, args, x_0, mu))
        x_0, _ = subgradient(A, b, mu, args, x_0, False)
        if mu > epsilon:
            mu = max(epsilon, gamma * mu)
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "Lasso_penalty")
    return (x_0, k, f) if output_f is True else (x_0, k)

'''
近似点梯度法
'''
def approximate_point_gradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, epsilon: Optional[float]=1e-4, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    A : NDArray
        m*n维数 参数矩阵
        
    b : NDArray
        m*1维数 参数矩阵
        
    mu : float
        正则化参数
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    args = a2m(args)
    values, _ = np.linalg.eig((A.T).dot(A))
    lambda_ma = max(values)
    if isinstance(lambda_ma, complex):
        tk = 1 / np.real(lambda_ma)
    else:
        tk = 1 / lambda_ma
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    f = []
    point = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0, mu))
        point.append(x_0)
        grad = np.array(res.subs(reps)).astype(DataType)
        x_0 = np.sign(x_0 - tk * grad[0]) * [max(i, 0) for i in np.abs(x_0 - tk * grad[0]) - tk * mu]
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            f.append(get_value(funcs, args, x_0, mu))
            break
    plot_iteration(f, draw, "Lasso_approximate_point_gradient")
    return (x_0, k, f) if output_f is True else (x_0, k)