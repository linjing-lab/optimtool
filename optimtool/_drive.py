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
from ._typing import SympyMutableDenseMatrix, List, IterPointType, NDArray, DataType, Tuple

def Q_k(eta: float, k: int) -> float:
    '''
    Parameters
    ----------
    eta : float
        常数
        
    k : int
        迭代次数
        

    Returns
    -------
    float
        常数
        
    '''
    assert k >= 0
    if k == 0:
        return 1.0
    else:
        return eta * Q_k(eta, k-1) + 1

def C_k(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, point: List[IterPointType], eta: float, k: int) -> DataType:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数
        
    point : List[IterPointType]
        当前迭代点
        
    eta : float
        常数
        
    k : int
        当前迭代次数
        

    Returns
    -------
    DataType
        常数
        
    '''
    assert k >= 0
    if k == 0:
        return np.array(funcs.subs(dict(zip(args, point[0])))).astype(DataType)
    else:
        return (1 / (Q_k(eta, k))) * (eta * Q_k(eta, k-1) * C_k(funcs, args, point, eta, k - 1) + np.array(funcs.subs(dict(zip(args, point[k])))).astype(DataType))

def get_f_delta_gradient(resv: NDArray, argsv: NDArray, mu: float, delta: float) -> DataType:
    '''
    Parameters
    ----------
    resv : NDArray
        当前梯度值
        
    argsv : NDArray
        当前参数值
        
    mu : float
        正则化参数
        
    delta : float
        常数
        

    Returns
    -------
    DataType
        当前梯度
        
    '''
    f = []
    for i, j in zip(resv, argsv):
        abs_args = np.abs(j)
        if abs_args > delta:
            if j > 0:
                f.append(i + mu * 1)
            elif j < 0:
                f.append(i - mu * 1)
        else:
            f.append(i + mu * (j / delta))
    return f[0]

def get_subgradient(resv: NDArray, argsv: NDArray, mu: float) -> DataType:
    '''
    Parameters
    ----------
    resv : NDArray
        当前梯度值
        
    argsv : NDArray
        当前参数值
        
    mu : float
        正则化参数
        

    Returns
    -------
    DataType
        当前次梯度
        
    '''
    f = []
    for i, j in zip(resv, argsv):
        if j > 0:
            f.append(i + mu * 1)
        elif j == 0:
            f.append(i + mu * (2 * np.random.random_sample() - 1))
        else:
            f.append(i - mu * 1)
    return f[0]

def CG_gradient(A: NDArray, b: NDArray, dk: NDArray, epsilon: float=1e-6, k: int=0) -> Tuple[NDArray,int]:
    '''
    Parameters
    ----------
    A : NDArray
        矩阵
        
    b : NDArray
        行向量
        
    dk : NDArray
        初始梯度下降方向（列向量）
        
    epsilon : float
        精度
        
    k : int
        迭代次数
        

    Returns
    -------
    Tuple[NDArray,int]
        当前梯度（行向量）, 迭代次数
        
    '''
    rk = b.T - A.dot(dk)
    pk = rk
    while 1:
        if np.linalg.norm(pk) < epsilon:
            break
        else:
            ak = (rk.T).dot(rk) / ((pk.T).dot(A)).dot(pk)
            dk = dk + ak * pk
            bk_down = (rk.T).dot(rk)
            rk = rk - ak * A.dot(pk)
            bk = (rk.T).dot(rk) / bk_down
            pk = rk + bk * pk
        k = k + 1
    return dk.reshape(1, -1), k

def L_BFGS_double_loop(q: NDArray, p: List[NDArray], s: List[NDArray], y: List[NDArray], m: int, k: int, Hkm: NDArray) -> NDArray:
    '''
    Parameters
    ----------
    q : NDArray
        初始梯度方向（行向量）
        
    p : List[NDArray]
        当前pk的列表
        
    s : List[NDArray]
        当前sk的列表
        
    y : List[NDArray]
        当前yk的列表
        
    m : int
        双循环阈值
        
    k : int
        迭代次数
        
    Hkm : NDArray
        双循环初始矩阵
        

    Returns
    -------
    NDArray
        当前梯度
        
    '''
    istart1 = max(0, k - 1)
    iend1 = max(0, k - m - 1)
    istart2 = max(0, k - m)
    iend2 = max(0, k)
    alpha = np.empty((k, 1))
    for i in range(istart1, iend1, -1):
        alphai = p[i] * s[i].dot(q.T)
        alpha[i] = alphai
        q = q - alphai * y[i]
    r = Hkm.dot(q.T)
    for i in range(istart2, iend2):
        beta = p[i] * y[i].dot(r)
        r = r + (alpha[i] - beta) * s[i].T
    return -r.reshape(1, -1)

def Eq_Sovle(sk: NDArray, pk: NDArray, delta: float):
    '''
    Parameters
    ----------
    sk : NDArray
        初始点
        
    pk : NDArray
        负梯度向量（行向量）
        
    delta : float
        搜索半径

    Returns
    -------
    float
        大于0的解

    '''
    m = sp.symbols("m", positive=True)
    r = (sk + m * pk)[0]
    sub = 0
    for i in r:
        sub += i**2
    h = sp.sqrt(sub) - delta
    mt = sp.solve(h)
    return mt[0]

def steihaug(sk: List[int], rk: NDArray, pk: NDArray, B: NDArray, delta: float, epsilon: float=1e-3, k: int=0) -> Tuple[NDArray,int]:
    '''
    Parameters
    ----------
    s0 : List[int]
        初始点列表
        
    rk : NDArray
        梯度向量（行向量）
        
    pk : NDArray
        负梯度向量（行向量）
        
    B : NDArray
        修正后的海瑟矩阵
        
    delta : float
        搜索半径
        
    epsilon : float
        精度
        
    k : int
        迭代次数
        

    Returns
    -------
    Tuple[NDArray,int]
        梯度，迭代次数
        
    '''
    s, r, p = [], [], []
    while 1:
        s.append(sk)
        r.append(rk)
        p.append(pk)
        pbp = (p[k].dot(B)).dot(p[k].T)
        if pbp <= 0:
            m = Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        alphak = np.linalg.norm(r[k])**2 / pbp
        sk = s[k] + alphak * p[k]
        if np.linalg.norm(sk) > delta:
            m = Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        rk = r[k] + alphak * (B.dot(p[k].T)).T
        if np.linalg.norm(rk) < epsilon * np.linalg.norm(r[0]):
            ans = sk
            break
        betak = np.linalg.norm(rk)**2 / np.linalg.norm(r[k])**2
        pk = - rk + betak * p[k]
        k = k + 1
    return ans.astype(DataType), k

def cons_unequal_L(cons_unequal: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, muk: NDArray, sigma: float, x_0: IterPointType) -> SympyMutableDenseMatrix:
    '''
    Parameters
    ----------
    cons_unequal : SympyMutableDenseMatrix
        当前不等式约束列表
        
    args : SympyMutableDenseMatrix
        参数列表
    
    muk : NDArray
        因子列表
        
    sigma : float
        常数
        
    x_0 : IterPointType
        当前迭代点
        

    Returns
    -------
    SympyMutableDenseMatrix
        加入因子约束后的不等式约束方程
        
    '''
    sub = 0
    for i in range(cons_unequal.shape[0]):
        cons = muk[i] / sigma + cons_unequal[i]
        con = sp.Matrix([cons])
        conv = np.array(con.subs(dict(zip(args, x_0)))).astype(DataType)
        if conv > 0:
            sub = sub + (cons**2 - (muk[i] / sigma)**2)
        else:
            sub = sub - (muk[i] / sigma)**2
    sub = sp.Matrix([sub])
    return sub

def v_k(cons_equal: SympyMutableDenseMatrix, cons_unequal: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, muk: NDArray, sigma: float, x_0: IterPointType) -> DataType:
    '''
    Parameters
    ----------
    cons_equal : SympyMutableDenseMatrix
        当前等式约束列表
    
    cons_unequal : SympyMutableDenseMatrix
        当前不等式约束列表
        
    args : SympyMutableDenseMatrix
        参数列表
    
    muk : NDArray
        因子列表
        
    sigma : float
        常数
        
    x_0 : IterPointType
        当前迭代点
        

    Returns
    -------
    DataType
        终止常数
        
    '''
    sub = 0
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(DataType)
    if cons_equal is not None:
        consv_equal = np.array(cons_equal.subs(reps)).astype(DataType)
        sub += (consv_equal.T).dot(consv_equal)
        for i in range(len_unequal):
            sub += (max(consv_unequal[i], - muk[i] / sigma))**2
    else:
        for i in range(len_unequal):
            sub += (max(consv_unequal[i], - muk[i] / sigma))**2
    return np.sqrt(sub)

def renew_mu_k(cons_unequal: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, muk: NDArray, sigma: float, x_0: IterPointType) -> NDArray:
    '''
    Parameters
    ----------
    cons_unequal : SympyMutableDenseMatrix
        当前不等式约束列表
        
    args : SympyMutableDenseMatrix
        参数列表
    
    muk : NDArray
        因子列表
        
    sigma : float
        常数
        
    x_0 : IterPointType
        当前迭代点
        

    Returns
    -------
    NDArray
        更新后的muk
        
    '''
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(DataType)
    for i in range(len_unequal):
        muk[i] = max(muk[i] + sigma * consv_unequal[i], 0)
    return muk