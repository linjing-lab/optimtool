import numpy as np
import sympy as sp
from functions.tools import function_f_x_k, function_plot_iteration
from optimtool.unconstrain import newton_quasi

# 二次罚函数法（不等式约束）
def penalty_quadratic(funcs, args, cons, x_0, draw=True, output_f=False, sigma=1, p=0.4, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
    epsilon : double
        迭代停机准则
        
    k : double
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert sigma > 0
    assert p > 0
    assert p < 1
    point = []
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(np.float64)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons.T * consv])
        x_0, _ = newton_quasi.L_BFGS(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_unequal") 
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# 内点罚函数法（不等式约束）
'''
保证点在定义域内
'''
# 分式
def penalty_interior_fraction(funcs, args, cons, x_0, draw=True, output_f=False, sigma=12, p=0.6, epsilon=1e-6, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
    epsilon : double
        迭代停机准则
        
    k : double
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert sigma > 0
    assert p > 0
    assert p < 1
    point = []
    f = []
    sub_pe = 0
    for i in cons:
        sub_pe += 1 / i
    sub_pe = sp.Matrix([sub_pe])
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = sp.Matrix([funcs - sigma * sub_pe])
        x_0, _ = newton_quasi.L_BFGS(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_interior_fraction")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
