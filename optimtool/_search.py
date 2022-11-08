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

__all__ = ['armijo', 'goldstein', 'wolfe', 'Grippo', 'ZhangHanger']

import numpy as np
from ._typing import Optional, List, NDArray, SympyMutableDenseMatrix, DataType, IterPointType

# Armijo线搜索准则
def armijo(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, d: NDArray, gamma: Optional[float]=0.5, c: Optional[float]=0.1) -> float:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点列表
        
    d : NDArray
        当前下降方向
        
    gamma : Optional[float]
        修正参数
        
    c : Optional[float]
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
    alpha = 1.0
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            break
        else:
            alpha = gamma * alpha
    return alpha

# Goldstein线搜索准则
def goldstein(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, d: NDArray, c: Optional[float]=0.1, alphas: Optional[float]=0, alphae: Optional[float]=10, t: Optional[float]=1.2, eps: Optional[float]=1e-3) -> float:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点列表
        
    d : NDArray
        当前下降方向
        
    alphas : Optional[float]
        起始搜索区间
        
    alphae : Optional[float]
        终止搜索区间
        
    t : Optional[float]
        扩大倍数参数
        
    eps : Optional[float]
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
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            if f1 >= f0 + (1 - c)*alpha*res0.dot(d.T):
                break
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
            break
    return alpha

# Wolfe线搜索准则
def wolfe(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, d: NDArray, c1: Optional[float]=0.3, c2: Optional[float]=0.5, alphas: Optional[float]=0, alphae: Optional[float]=2, eps: Optional[float]=1e-3) -> float:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点列表
        
    d : NDArray
        当前下降方向
        
    c1 : Optional[float]
        常数
        
    c2 : Optional[float]
        常数
        
    alphas : Optional[float]
        起始搜索区间
        
    alphae : Optional[float]
        终止搜索区间
        
    eps : Optional[float]
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
    f0 = np.array(funcs.subs(reps)).astype(DataType)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        if f1 <= f0 + c1*alpha*res0.dot(d.T):
            res1 = np.array(res.subs(dict(zip(args, x)))).astype(DataType)
            if res1.dot(d.T) >= c2*res0.dot(d.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas-alphae) < eps:
            break
    return alpha

# 非单调线搜索准则之Grippo（一般与Barzilar Borwein梯度下降法配合使用）
def Grippo(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, d: NDArray, k: int, point: List[IterPointType], c1: float, beta: float, alpha: float, M: Optional[int]=20) -> float:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点列表
        
    d : NDArray
        当前下降方向
        
    k : int
        当前迭代次数
        
    point : List[IterPointType]
        当前迭代点列表
    
    c1 : float
        常数
        
    beta : float
        修正参数
        
    alpha : float
        初始步长

    M : int
        阈值
        

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
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        fk = - np.inf
        for j in range(min(k, M) + 1):
            fk = max(fk, np.array(funcs.subs(dict(zip(args, point[k-j])))).astype(DataType))
        if f1 <= fk + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha

# 非单调线搜索准则之ZhangHanger（一般与程序配套使用）
def ZhangHanger(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, d: NDArray, k: int, point: List[IterPointType], c1: float, beta: float, alpha: float, eta: Optional[float]=0.6) -> float:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点列表
        
    d : NDArray
        当前下降方向
        
    k : int
        当前迭代次数
        
    point : List[IterPointType]
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
    from ._utils import C_k
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(DataType)
    while 1:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(DataType)
        Ck = C_k(funcs, args, point, eta, k)
        if f1 <= Ck + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha