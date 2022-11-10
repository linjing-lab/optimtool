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
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

# 二次罚函数法（等式约束）
def penalty_quadratice(funcs: FuncArray, args: FuncArray, cons: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="gradient_descent", sigma: float=10.0, p: float=2.0, epsilon: float=1e-4, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons : FuncArray
        等式参数约束列表
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : str
        无约束优化方法内核
        
    sigma : float
        罚函数因子
        
    p : float
        修正参数
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert sigma > 0
    assert p > 1
    from .._kernel import kernel, barzilar_borwein, CG, L_BFGS, steihaug_CG
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search = eval(kernel(method))
    point = []
    sig = sp.symbols("sig")
    pen = funcs + (sig / 2) * cons.T * cons
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        pe = pen.subs(sig, sigma)
        x_0, _ = search(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            break
        sigma = p * sigma
    plot_iteration(f, draw, "penalty_quadratic_equal")     
    return (x_0, k, f) if output_f is True else (x_0, k)

# 增广拉格朗日函数乘子法（等式约束）
def lagrange_augmentede(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="gradient_descent", lamk: float=6, sigma: float=10, p: float=2, etak: float=1e-4, epsilon: float=1e-6, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons : FuncArray
        等式参数约束列表
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : str
        无约束优化方法内核
        
    lamk : float
        因子
        
    sigma : float
        罚函数因子
        
    p : float
        修正参数
        
    etak : float
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
    assert sigma > 0
    assert p > 1
    from .._kernel import kernel, barzilar_borwein, CG, L_BFGS, steihaug_CG
    search = eval(kernel(method))
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    f = []
    lamk = np.array([lamk for i in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while 1:
        L = sp.Matrix([funcs + (sigma / 2) * cons.T * cons + cons.T * lamk])
        f.append(get_value(funcs, args, x_0))
        x_0, _ = search(L, args, tuple(x_0), draw=False, epsilon=etak)
        k = k + 1
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(DataType)
        if np.linalg.norm(consv) <= epsilon:
            f.append(get_value(funcs, args, x_0))
            break
        lamk = lamk + sigma * consv
        sigma = p * sigma
    plot_iteration(f, draw, "lagrange_augmented_equal")
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [penalty_quadratice, lagrange_augmentede]