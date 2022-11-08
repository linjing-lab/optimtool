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

__all__ = ['solve', 'steepest', 'barzilar_borwein']

import numpy as np
import sympy as sp
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t

from .._typing import FuncArray, ArgArray, PointArray, Optional, DataType, OutputType

# 梯度下降法
def solve(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, epsilon: Optional[float]=1e-10, k: Optional[int]=0) -> OutputType:
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
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    res = funcs.jacobian(args)
    m = sp.symbols("m")
    arg = sp.Matrix([m])
    fx = []
    while 1:
        reps = dict(zip(args, x_0))
        fx.append(get_value(funcs, args, x_0))
        dk = -np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            xt = x_0 + m * dk[0]
            f = funcs.subs(dict(zip(args, xt)))
            h = f.jacobian(arg)
            mt = sp.solve(h)
            x_0 = (x_0 + mt[m] * dk[0]).astype(DataType)
            k = k + 1
        else:
            break
    plot_iteration(fx, draw, "gradient_descent_solve")
    return (x_0, k, fx) if output_f is True else (x_0, k)

# 最速下降法
def steepest(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, method: Optional[str]="wolfe", epsilon: Optional[float]=1e-10, k: Optional[int]=0) -> OutputType:
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
        非精确线搜索方法
        
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
    search = eval(method)
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    res = funcs.jacobian(args)
    fx = []
    while 1:
        reps = dict(zip(args, x_0))
        fx.append(get_value(funcs, args, x_0))
        dk = -np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    plot_iteration(fx, draw, "gradient_descent_steepest")
    return (x_0, k, fx) if output_f is True else (x_0, k)
    
# Barzilar Borwein梯度下降法
def barzilar_borwein(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: Optional[bool]=True, output_f: Optional[bool]=False, method: Optional[str]="Grippo", c1: Optional[float]=0.6, beta: Optional[float]=0.6, alpha: Optional[float]=1, epsilon: Optional[float]=1e-10, k: Optional[int]=0) -> OutputType:
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
        非单调线搜索方法："Grippo"与"ZhangHanger"
        
    M : Optional[int]
        阈值
        
    c1 : Optional[float]
        常数
        
    beta : Optional[float]
        常数
        
    alpha : Optional[float]
        初始步长
        
    epsilon : Optional[float]
        迭代停机准则
        
    k : Optional[int]
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    from .._search import Grippo, ZhangHanger
    search = eval(method)
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    res = funcs.jacobian(args)
    point, f = [], []
    while 1:
        point.append(x_0)
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        dk = - np.array(res.subs(reps)).astype(DataType)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk, k, point, c1, beta, alpha)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = delta
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(DataType) + dk
            if sk.dot(yk.T) != 0:
                alpha = sk.dot(sk.T) / sk.dot(yk.T)
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "gradient_descent_barzilar_borwein_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)