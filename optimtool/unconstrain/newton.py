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

import numpy as np
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t, h2h

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

def classic(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, epsilon: float=1e-10, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    res = funcs.jacobian(args) # gradient
    hes, f = res.jacobian(args), []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hessian = np.array(hes.subs(reps)).astype(DataType)
        dk = -np.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            x_0 += dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_classic")        
    return (x_0, k, f) if output_f is True else (x_0, k)

def modified(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="wolfe", m: int=20, epsilon: float=1e-10, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表

    method : str
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res = funcs.jacobian(args) # graident
    hes = res.jacobian(args) # hessian
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hessian = np.array(hes.subs(reps)).astype(DataType) # hessian: from `object` to `float`
        hessian = h2h(hessian)
        dk = -np.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "newton_modified_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

def CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-6, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表

    method : str
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        f
    '''
    from .._drive import CG_gradient
    from .._kernel import linear_search
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    search, f = linear_search(method), []
    res = funcs.jacobian(args) # gradient
    hes, dk0 = res.jacobian(args), np.zeros((args.shape[0], 1)) # hessian and initial dk
    while 1:
        reps = dict(zip(args, x_0))
        f.append(get_value(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(DataType)
        hess = np.array(hes.subs(reps)).astype(DataType)
        dk, _ = CG_gradient(hess, -gradient, dk0)
        if np.linalg.norm(dk) >= epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "newton_CG_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [classic, modified, CG]