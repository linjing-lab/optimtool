__all__ = ['armijo', 'goldstein', 'wolfe', 'nonmonotonic_Grippo', 'nonmonotonic_ZhangHanger']

import numpy as np
from functions.tools import C_k

# Armijo线搜索准则
def armijo(funcs, args, x_0, d, gamma=0.5, c=0.1):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    d : numpy.array
        当前下降方向
        
    gamma : float
        修正参数
        
    c : float
        常数
        

    Returns
    -------
    float
        最优步长
        
    '''
    assert gamma > 0
    assert gamma < 1
    assert c > 0
    assert c < 1
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            break
        else:
            alpha = gamma * alpha
    return alpha

# Goldstein线搜索准则
def goldstein(funcs, args, x_0, d, c=0.1, alphas=0, alphae=10, t=1.2, eps=1e-3):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    d : numpy.array
        当前下降方向
        
    alphas : float
        起始搜索区间
        
    alphae : float
        终止搜索区间
        
    t : float
        扩大倍数参数
        
    eps : float
        终止参数
        

    Returns
    -------
    float
        最优步长
        
    '''
    assert c > 0
    assert c < 0.5
    assert alphas < alphae
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            if f1 >= f0 + (1 - c)*alpha*res0.dot(d.T):
                break;
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
                if alphae < np.inf:
                    alpha = 0.5 * (alphas + alphae)
                else:
                    alpha = t * alpha
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas-alphae) < eps:
            break;
    return alpha

# Wolfe线搜索准则
def wolfe(funcs, args, x_0, d, c1=0.3, c2=0.5, alphas=0, alphae=2, eps=1e-3):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    d : numpy.array
        当前下降方向
        
    c1 : float
        常数
        
    c2 : float
        常数
        
    alphas : float
        起始搜索区间
        
    alphae : float
        终止搜索区间
        
    eps : float
        终止参数
        

    Returns
    -------
    float
        最优步长
        
    '''
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas < alphae
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c1*alpha*res0.dot(d.T):
            res1 = np.array(res.subs(dict(zip(args, x)))).astype(np.float64)
            if res1.dot(d.T) >= c2*res0.dot(d.T):
                break;
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas-alphae) < eps:
            break;
    return alpha

# 非单调线搜索准则之Grippo（一般与Barzilar Borwein梯度下降法配合使用）
def nonmonotonic_Grippo(funcs, args, x_0, d, k, point, M, c1, beta, alpha):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    d : numpy.array
        当前下降方向
        
    k : int
        当前迭代次数
        
    point : list
        当前迭代点列表
        
    M : int
        阈值
    
    c1 : float
        常数
        
    beta : float
        修正参数
        
    alpha : float
        初始步长
        

    Returns
    -------
    float
        最优步长
        
    '''
    assert M >= 0
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        fk = - np.inf
        for j in range(min(k, M) + 1):
            fk = max(fk, np.array(funcs.subs(dict(zip(args, point[k-j])))).astype(np.float64))
        if f1 <= fk + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha

# 非单调线搜索准则之ZhangHanger（一般与程序配套使用）
def nonmonotonic_ZhangHanger(funcs, args, x_0, d, k, point, c1, beta, alpha, eta=0.6):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    d : numpy.array
        当前下降方向
        
    k : int
        当前迭代次数
        
    point : list
        当前迭代点列表
    
    c1 : float
        常数
        
    beta : float
        修正参数
        
    alpha : float
        初始步长
        
    eta : float
        常数
        

    Returns
    -------
    float
        最优步长
        
    '''
    assert eta > 0
    assert eta < 1
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        Ck = C_k(funcs, args, point, eta, k)
        if f1 <= Ck + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha