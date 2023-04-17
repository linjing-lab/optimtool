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

# 二次罚函数法（混合约束）
def penalty_quadraticm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10, p: float=0.6, epsilon: float=1e-10, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons_equal : FuncArray
        等式参数约束列表
        
    cons_unequal : FuncArray
        不等式参数约束列表
        
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
    assert p > 0
    from .._kernel import kernel, barzilar_borwein, modified, L_BFGS, steihaug_CG
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search = eval(kernel(method))
    f = []
    point = []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons_unequal.subs(reps)).astype(DataType)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons_unequal.T * consv + (sigma / 2) * cons_equal.T * cons_equal])
        x_0, _ = search(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            break
        sigma = p * sigma
    plot_iteration(f, draw, "penalty_quadratic_mixequal")
    return (x_0, k, f) if output_f is True else (x_0, k)

# 精确罚函数法-l1罚函数法 （混合约束）
def penalty_L1(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=1, p: float=0.6, epsilon: float=1e-10, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons_equal : FuncArray
        等式参数约束列表
        
    cons_unequal : FuncArray
        不等式参数约束列表
        
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
    assert p > 0
    from .._kernel import kernel, barzilar_borwein, modified, L_BFGS, steihaug_CG
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search = eval(kernel(method))
    point = []
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv_unequal = np.array(cons_unequal.subs(reps)).astype(DataType)
        consv_unequal = np.where(consv_unequal <= 0, consv_unequal, 1)
        consv_unequal = np.where(consv_unequal > 0, consv_unequal, 0)
        consv_equal = np.array(cons_equal.subs(reps)).astype(DataType)
        consv_equal = np.where(consv_equal <= 0, consv_equal, 1)
        consv_equal = np.where(consv_equal > 0, consv_equal, -1)
        pe = sp.Matrix([funcs + sigma * cons_unequal.T * consv_unequal + sigma * cons_equal.T * consv_equal])
        x_0, _ = search(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            break
        sigma = p * sigma
    plot_iteration(f, draw, "penalty_L1")
    return (x_0, k, f) if output_f is True else (x_0, k)

# 增广拉格朗日函数法（混合约束）
def lagrange_augmentedm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", lamk: float=6, muk: float=10, sigma: float=8, alpha: float=0.5, beta: float=0.7, p: float=2, eta: float=1e-3, epsilon: float=1e-4, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons_equal : FuncArray
        等式参数约束列表
        
    cons_unequal : FuncArray
        不等式参数约束列表
        
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
    
    muk : float
        因子
    
    sigma : float
        罚函数因子
    
    alpha : float
        初始步长
    
    beta : float
        修正参数
    
    p : float
        修正参数
    
    eta : float
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
    assert alpha > 0 
    assert alpha <= beta
    assert beta < 1
    from .._kernel import kernel, barzilar_borwein, modified, L_BFGS, steihaug_CG
    from .._drive import cons_unequal_L, v_k, renew_mu_k
    funcs, args, cons_equal, cons_unequal, x_0 = f2m(funcs), a2m(args), f2m(cons_equal), f2m(cons_unequal), p2t(x_0)
    search = eval(kernel(method))
    f = []
    lamk = np.array([lamk for i in range(cons_equal.shape[0])]).reshape(cons_equal.shape[0], 1)
    muk = np.array([muk for i in range(cons_unequal.shape[0])]).reshape(cons_unequal.shape[0], 1)
    while 1:
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = cons_unequal_L(cons_unequal, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * (cons_equal.T * cons_equal + cons_uneuqal_modifyed) + cons_equal.T * lamk])
        f.append(get_value(funcs, args, x_0))
        x_0, _ = search(L, args, tuple(x_0), draw=False, epsilon=epsilonk)
        k = k + 1
        vkx = v_k(cons_equal, cons_unequal, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)) <= eta):
                f.append(get_value(funcs, args, x_0))
                break
            else:
                lamk = lamk + sigma * np.array(cons_equal.subs(dict(zip(args, x_0)))).astype(DataType)
                muk = renew_mu_k(cons_unequal, args, muk, sigma, x_0)
                etak = etak / sigma
                epsilonk = epsilonk / sigma**beta
        else:
            sigma = p * sigma
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    plot_iteration(f, draw, "lagrange_augmented_mixequal")
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [penalty_quadraticm, penalty_L1, lagrange_augmentedm]