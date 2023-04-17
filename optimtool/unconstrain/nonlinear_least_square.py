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
import sympy as sp
from .._convert import f2m, a2m, p2t, h2h
from .._utils import get_value, plot_iteration

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

# 高斯-牛顿法（非线性最小二乘问题）
def gauss_newton(funcr: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType:
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
    from .._search import armijo, goldstein, wolfe
    funcr, args, x_0 = f2m(funcr), a2m(args), p2t(x_0)
    assert funcr.shape[0] > 1 and funcr.shape[1] ==1 and args.shape[0] == len(x_0)
    search, f = eval(method), []
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(DataType)
        f.append(get_value(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(DataType)
        q, r = np.linalg.qr(jk)
        dk = np.linalg.inv(r).dot(-(q.T).dot(rk)).reshape(1,-1)
        if np.linalg.norm(dk) > epsilon:
            alpha = search(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "nonlinear_least_square_gauss_newton_" + method)
    return (x_0, k, f) if output_f is True else (x_0, k)

# levenberg marquardt方法
def levenberg_marquardt(funcr: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, m: float=100, lamk: float=1, eta: float=0.2, p1: float=0.4, p2: float=0.9, gamma1: float=0.7, gamma2: float=1.3, epsilon: float=1e-10, k: int=0) -> OutputType:
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
        
    lamk : float
        修正常数
        
    eta : float
        常数
        
    p1 : float 
        常数
        
    p2 : float
        常数
        
    gamma1 : float
        常数
        
    gamma2 : float
        常数
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert eta >= 0
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    from .._drive import CG_gradient
    funcr, args, x_0 = f2m(funcr), a2m(args), p2t(x_0)
    assert funcr.shape[0] > 1 and funcr.shape[1] ==1 and args.shape[0] == len(x_0)
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    resf = funcs.jacobian(args)
    hess = resf.jacobian(args)
    dk0 = np.zeros((args.shape[0], 1))
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(DataType)
        f.append(get_value(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(DataType)
        dk, _ = CG_gradient((jk.T).dot(jk) + lamk, -((jk.T).dot(rk)).reshape(1, -1), dk0)
        pk_up = np.array(funcs.subs(reps)).astype(DataType) - np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(DataType)
        grad_f = np.array(resf.subs(reps)).astype(DataType)
        hess_f = np.array(hess.subs(reps)).astype(DataType)
        hess_f = h2h(hess_f)
        pk_down = -(grad_f.dot(dk.T) + 0.5*((dk.dot(hess_f)).dot(dk.T)))
        pk = pk_up / pk_down
        if np.linalg.norm(dk) >= epsilon:
            if pk < p1:
                lamk = gamma2 * lamk
            else:
                if pk > p2:
                    lamk = gamma1 * lamk
                else:
                    lamk = lamk
            if pk > eta:
                x_0 = x_0 + dk[0]
            else:
                x_0 = x_0
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "nonlinear_least_square_levenberg_marquardt")        
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [gauss_newton, levenberg_marquardt]