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

import sympy as sp
import numpy as np
from .._utils import get_value, plot_iteration
from .._convert import f2m, a2m, p2t

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

# 二次罚函数法（不等式约束）
def penalty_quadraticu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10, p: float=0.4, epsilon: float=1e-10, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    cons : FuncArray
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
    assert p < 1
    from .._kernel import kernel, barzilar_borwein, modified, L_BFGS, steihaug_CG
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, point, f = eval(kernel(method)), [], []
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(DataType)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons.T * consv])
        x_0, _ = search(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            break
        sigma = p * sigma
    plot_iteration(f, draw, "penalty_quadratic_unequal") 
    return (x_0, k, f) if output_f is True else (x_0, k)

# 内点罚函数法（不等式约束）
'''
保证点在定义域内
'''

# 分式
def penalty_interior_fraction(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=12, p: float=0.6, epsilon: float=1e-6, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    cons : FuncArray
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
    assert p < 1
    from .._kernel import kernel, barzilar_borwein, modified, L_BFGS, steihaug_CG
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, point, f = eval(kernel(method)), [], []
    sub_pe = 0
    for i in cons:
        sub_pe += 1 / i
    sub_pe = sp.Matrix([sub_pe])
    while 1:
        point.append(np.array(x_0))
        f.append(get_value(funcs, args, x_0))
        pe = sp.Matrix([funcs - sigma * sub_pe])
        x_0, _ = search(pe, args, tuple(x_0), draw=False)
        k = k + 1
        sigma = p * sigma
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(get_value(funcs, args, x_0))
            break
    plot_iteration(f, draw, "penalty_interior_fraction")
    return (x_0, k, f) if output_f is True else (x_0, k)
    
# 增广拉格朗日函数法（不等式约束）
def lagrange_augmentedu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, draw: bool=True, output_f: bool=False, method: str="newton", muk: float=10, sigma: float=8, alpha: float=0.2, beta: float=0.7, p: float=2, eta: float=1e-1, epsilon: float=1e-4, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数列表
        
    cons : FuncArray
        不等式参数约束列表
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : str
        无约束优化方法内核
    
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
    from .._drive import cons_unequal_L, renew_mu_k, v_k
    funcs, args, x_0, cons = f2m(funcs), a2m(args), p2t(x_0), f2m(cons)
    search, f = eval(kernel(method)), []
    muk = np.array([muk for i in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while 1:
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = cons_unequal_L(cons, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * cons_uneuqal_modifyed])
        f.append(get_value(funcs, args, x_0))
        x_0, _ = search(L, args, x_0, draw=False, epsilon=epsilonk)
        k = k + 1
        vkx = v_k(None, cons, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(DataType)) <= eta):
                f.append(get_value(funcs, args, x_0))
                break
            else:
                muk = renew_mu_k(cons, args, muk, sigma, x_0)
                etak = etak / sigma
                epsilonk = epsilonk / sigma**beta
        else:
            sigma = p * sigma
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    plot_iteration(f, draw, "lagrange_augmented_unequal")
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [penalty_quadraticu, penalty_interior_fraction, lagrange_augmentedu]