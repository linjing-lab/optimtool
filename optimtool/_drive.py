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

from .base import np, sp
from ._typing import SympyMutableDenseMatrix, List, IterPointType, NDArray, DataType

def Q_k(eta: float, 
        k: int) -> float:
    assert k >= 0
    return 1.0 if k == 0 else eta * Q_k(eta, k-1) + 1

def C_k(funcs: SympyMutableDenseMatrix, 
        args: SympyMutableDenseMatrix, 
        point: List[IterPointType], 
        eta: float, 
        k: int) -> DataType:
    assert k >= 0
    return np.array(funcs.subs(dict(zip(args, point[0])))).astype(DataType) if k == 0 else (1 / (Q_k(eta, k))) * (eta * Q_k(eta, k-1) * C_k(funcs, args, point, eta, k - 1) + np.array(funcs.subs(dict(zip(args, point[k])))).astype(DataType))

def get_f_delta_gradient(resv: NDArray, 
                         argsv: NDArray, 
                         mu: float, 
                         delta: float) -> DataType:
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

def get_subgradient(resv: NDArray, 
                    argsv: NDArray, 
                    mu: float) -> DataType:
    f = []
    for i, j in zip(resv, argsv):
        if j > 0:
            f.append(i + mu * 1)
        elif j == 0:
            f.append(i + mu * (2 * np.random.random_sample() - 1))
        else:
            f.append(i - mu * 1)
    return f[0]

def conjugate(A: NDArray, 
              b: NDArray, 
              dk: NDArray, 
              eps: float) -> NDArray:
    rk = b.T - A.dot(dk)
    pk = rk
    while 1:
        if np.linalg.norm(pk) < eps:
            break
        else:
            ak = (rk.T).dot(rk) / ((pk.T).dot(A)).dot(pk)
            dk = dk + ak * pk
            bk_down = (rk.T).dot(rk)
            rk = rk - ak * A.dot(pk)
            bk = (rk.T).dot(rk) / bk_down
            pk = rk + bk * pk
    return dk

def double_loop(q: NDArray, 
                p: List[NDArray], 
                s: List[NDArray], 
                y: List[NDArray], 
                m: int, 
                k: int, 
                Hkm: NDArray) -> NDArray:
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
    return r

def Eq_Sovle(sk: NDArray, 
             pk: NDArray, 
             delta: float):
    m = sp.symbols("m", positive=True)
    r = (sk + m * pk)[0]
    sub = 0
    for i in r:
        sub += i**2
    h = sp.sqrt(sub) - delta
    mt = sp.solve(h)
    return mt[0]

def steihaug(sk: List[int], 
             rk: NDArray, 
             pk: NDArray, 
             B: NDArray, 
             delta: float, 
             epsilon: float=1e-3,
             k: int=0) -> NDArray:
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
    return ans

def cons_unequal_L(cons_unequal: SympyMutableDenseMatrix, 
                   args: SympyMutableDenseMatrix, 
                   muk: NDArray, 
                   sigma: float, 
                   x_0: IterPointType) -> SympyMutableDenseMatrix:
    sub = 0
    for i in range(cons_unequal.shape[0]):
        cons = muk[i] / sigma + cons_unequal[i]
        con = sp.Matrix([cons])
        conv = np.array(con.subs(dict(zip(args, x_0)))).astype(DataType)
        if conv > 0:
            sub = sub + (cons**2 - (muk[i] / sigma)**2)
        else:
            sub = sub - (muk[i] / sigma)**2
    return sp.Matrix([sub])

def v_k(cons_equal: SympyMutableDenseMatrix, 
        cons_unequal: SympyMutableDenseMatrix, 
        args: SympyMutableDenseMatrix, 
        muk: NDArray, 
        sigma: float, 
        x_0: IterPointType) -> DataType:
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

def renew_mu_k(cons_unequal: SympyMutableDenseMatrix, 
               args: SympyMutableDenseMatrix, 
               muk: NDArray, 
               sigma: float, 
               x_0: IterPointType) -> NDArray:
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(DataType)
    for i in range(len_unequal):
        muk[i] = max(muk[i] + sigma * consv_unequal[i], 0)
    return muk

def gammak(k: int) -> DataType:
    assert k >= 0
    return 1 if k == 0 else 2 / (1 + np.sqrt(1 + 4 / gammak(k-1)))