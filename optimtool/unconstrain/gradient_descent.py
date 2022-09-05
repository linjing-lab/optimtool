__all__ = ['solve', 'steepest', 'barzilar_borwein']

import numpy as np
import sympy as sp
from ..functions.tools import f_x_k, plot_iteration, data_convert
from ..functions.linear_search import armijo, goldstein, wolfe, nonmonotonic_Grippo, nonmonotonic_ZhangHanger

# 梯度下降法
def solve(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
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
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    funcs, args, _, _ = data_convert(funcs, args)
    res = funcs.jacobian(args)
    m = sp.symbols("m")
    arg = sp.Matrix([m])
    fx = []
    while 1:
        reps = dict(zip(args, x_0))
        fx.append(f_x_k(funcs, args, x_0))
        dk = -np.array(res.subs(reps)).astype(np.float64)
        if np.linalg.norm(dk) >= epsilon:
            xt = x_0 + m * dk[0]
            f = funcs.subs(dict(zip(args, xt)))
            h = f.jacobian(arg)
            mt = sp.solve(h)
            x_0 = (x_0 + mt[m] * dk[0]).astype(np.float64)
            k = k + 1
        else:
            break
    plot_iteration(fx, draw, "gradient_descent_solve")
    return x_0, k, fx if output_f is True else x_0, k

# 最速下降法
def steepest(funcs, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        非精确线搜索方法
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    funcs, args, _, _ = data_convert(funcs, args)
    res = funcs.jacobian(args)
    fx = []
    while 1:
        reps = dict(zip(args, x_0))
        fx.append(f_x_k(funcs, args, x_0))
        dk = -np.array(res.subs(reps)).astype(np.float64)
        if np.linalg.norm(dk) >= epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    plot_iteration(fx, draw, "gradient_descent_steepest")
    return x_0, k, fx if output_f is True else x_0, k
    
# Barzilar Borwein梯度下降法
def barzilar_borwein(funcs, args, x_0, draw=True, output_f=False, method="grippo", M=20, c1=0.6, beta=0.6, alpha=1, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        非单调线搜索方法："grippo"与"ZhangHanger"
        
    M : int
        阈值
        
    c1 : float
        常数
        
    beta : float
        常数
        
    alpha : float
        初始步长
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert M >= 0
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    funcs, args, _, _ = data_convert(funcs, args)
    res = funcs.jacobian(args)
    point = []
    f = []
    while 1:
        point.append(x_0)
        reps = dict(zip(args, x_0))
        f.append(f_x_k(funcs, args, x_0))
        dk = - np.array(res.subs(reps)).astype(np.float64)
        if np.linalg.norm(dk) >= epsilon:
            if method == "grippo":
                alpha = nonmonotonic_Grippo(funcs, args, x_0, dk, k, point, M, c1, beta, alpha)
            if method == "ZhangHanger":
                alpha = nonmonotonic_ZhangHanger(funcs, args, x_0, dk, k, point, c1, beta, alpha)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = delta
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) + dk
            if sk.dot(yk.T) != 0:
                alpha = sk.dot(sk.T) / sk.dot(yk.T)
            k = k + 1
        else:
            break
    plot_iteration(f, draw, "gradient_descent_barzilar_borwein_" + method)
    return x_0, k, f if output_f is True else x_0, k