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

__all__ = ['bfgs', 'dfp', 'L_BFGS']

import numpy as np
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t, h2h

from .._typing import FuncArray, ArgArray, PointArray, Optional, OutputType, DataType

# BFGS拟牛顿法
def bfgs(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, method: Optional[str]="wolfe", m: Optional[float]=20, epsilon: Optional[float]=1e-10, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    method : Optional[str]
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : Optional[float]
        海瑟矩阵条件数阈值
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._search import armijo, goldstein, wolfe
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    search = eval(method)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(DataType)
    hess = h2h(hess, m)
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(DataType)
        dk = - np.linalg.inv(hess).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) - np.array(res.subs(reps)).astype(DataType)
            if yk.all != 0:
                hess = hess + (yk.T).dot(yk) / sk.dot(yk.T) - (hess.dot(sk.T)).dot((hess.dot(sk.T)).T) / sk.dot((hess.dot(sk.T)))
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "newton_quasi_bfgs_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

# DFP拟牛顿法
def dfp(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, method: Optional[str]="wolfe", m: Optional[float]=20, epsilon: Optional[float]=1e-4, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    method : Optional[str]
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : Optional[float]
        海瑟矩阵条件数阈值
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._search import armijo, goldstein, wolfe
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    search = eval(method)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(DataType)
    hess = h2h(hess, m)
    hessi = np.linalg.inv(hess)
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(DataType)
        dk = - hessi.dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) - np.array(res.subs(reps)).astype(DataType)
            if yk.all != 0:
                hessi = hessi - (hessi.dot(yk.T)).dot((hessi.dot(yk.T)).T) / yk.dot(hessi.dot(yk.T)) + (sk.T).dot(sk) / yk.dot(sk.T)
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "newton_quasi_dfp_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

# L_BFGS方法
def L_BFGS(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, method: Optional[str]="wolfe", m: Optional[float]=6, epsilon: Optional[float]=1e-10, k: Optional[int]=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : Optional[bool]
        绘图接口参数
        
    output_f : Optional[bool]
        输出迭代函数值列表
        
    method : Optional[str]
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : Optional[float]
        海瑟矩阵条件数阈值
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._search import armijo, goldstein, wolfe
    from .._drive import L_BFGS_double_loop
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    search = eval(method)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    l = hes.shape[0]
    f, s, y, p = [], [], [], []
    gamma = []
    gamma.append(1)
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        Hkm = gamma[k] * np.identity(l)
        grad = np.array(res.subs(reps)).astype(DataType)
        dk = L_BFGS_double_loop(grad, p, s, y, m, k, Hkm)
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