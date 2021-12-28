import numpy as np
import sympy as sp
from functions.tools import function_f_x_k, function_plot_iteration
from optimtool.unconstrain import newton_quasi

# 二次罚函数法（等式约束）
def penalty_quadratic(funcs, args, cons, x_0, draw=True, output_f=False, sigma=10, p=2, epsilon=1e-4, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        等式参数约束列表
        
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
    assert p > 1
    point = []
    sig = sp.symbols("sig")
    pen = funcs + (sig / 2) * cons.T * cons
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = pen.subs(sig, sigma)
        x_0, _ = newton_quasi.L_BFGS(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_equal")     
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# 增广拉格朗日函数乘子法（等式约束）
def lagrange_augmented(funcs, args, cons, x_0, draw=True, output_f=False, lamk=6, sigma=10, p=2, etak=1e-4, epsilon=1e-6, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    lamk : double
        因子
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
    etak : double
        常数
        
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
    assert p > 1
    f = []
    lamk = np.array([lamk for i in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while True:
        L = sp.Matrix([funcs + (sigma / 2) * cons.T * cons + cons.T * lamk])
        f.append(function_f_x_k(funcs, args, x_0))
        x_0, _ = newton_quasi.L_BFGS(L, args, x_0, draw=False, epsilon=etak)
        k = k + 1
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(np.float64)
        if np.linalg.norm(consv) <= epsilon:
            f.append(function_f_x_k(funcs, args, x_0))
            break
        lamk = lamk + sigma * consv
        sigma = p * sigma
    function_plot_iteration(f, draw, "lagrange_augmented_equal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k