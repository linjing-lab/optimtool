def function_f_x_k(funcs, args, x_0, mu=None):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list or tuple
        初始迭代点列表（或元组）
        
    mu : float
        正则化参数
        

    Returns
    -------
    float
        迭代函数值
        
    '''
    import numpy as np
    funcsv = np.array(funcs.subs(dict(zip(args, x_0)))).astype(np.float64)
    if mu is not None:
        for i in x_0:
            funcsv += mu * np.abs(i)
    return funcsv[0][0]

def function_plot_iteration(f, draw, method):
    '''
    Parameters
    ----------
    f : list
        迭代函数值列表
        
    draw : bool
        绘图参数
        
    method : string
        最优化方法
        

    Returns
    -------
    None
        
    '''
    import matplotlib.pyplot as plt
    if draw is True:
        plt.plot([i for i in range(len(f))], f, marker='o', c="maroon", ls='--')
        plt.xlabel("$k$")
        plt.ylabel("$f(x_k)$")
        plt.title(method)
        plt.show()
    return None

def function_Q_k(eta, k):
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
        return 1
    else:
        return eta * function_Q_k(eta, k-1) + 1

def function_C_k(funcs, args, point, eta, k):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    point : list
        当前迭代点列表
        
    eta : float
        常数
        
    k : int
        当前迭代次数
        

    Returns
    -------
    float
        常数
        
    '''
    import numpy as np
    assert k >= 0
    if k == 0:
        return np.array(funcs.subs(dict(zip(args, point[0])))).astype(np.float64)
    else:
        return (1 / (function_Q_k(eta, k))) * (eta * function_Q_k(eta, k-1) * function_C_k(funcs, args, point, eta, k - 1) + np.array(funcs.subs(dict(zip(args, point[k])))).astype(np.float64))

def function_get_f_delta_gradient(resv, argsv, mu, delta):
    '''
    Parameters
    ----------
    resv : numpy.array
        当前梯度值
        
    argsv : numpy.array
        当前参数值
        
    mu : float
        正则化参数
        
    delta : float
        常数
        

    Returns
    -------
    float
        当前梯度
        
    '''
    import numpy as np
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

def function_get_subgradient(resv, argsv, mu):
    '''
    Parameters
    ----------
    resv : numpy.array
        当前梯度值
        
    argsv : numpy.array
        当前参数值
        
    mu : float
        正则化参数
        

    Returns
    -------
    float
        当前次梯度
        
    '''
    import numpy as np
    f = []
    for i, j in zip(resv, argsv):
        if j > 0:
            f.append(i + mu * 1)
        elif j == 0:
            f.append(i + mu * (2 * np.random.random_sample() - 1))
        else:
            f.append(i - mu * 1)
    return f[0]

def function_modify_hessian(hessian, m, pk=1):
    '''
    Parameters
    ----------
    hessian : numpy.array
        未修正的海瑟矩阵值
        
    m : float
        条件数阈值
        
    pk : int
        常数
        

    Returns
    -------
    numpy.array
        修正后的海瑟矩阵
        
    '''
    import numpy as np
    l = hessian.shape[0]
    while True:
        values, _ = np.linalg.eig(hessian)
        flag = (all(values) > 0) & (np.linalg.cond(hessian) <= m)
        if flag:
            break
        else:
            hessian = hessian + pk * np.identity(l)
            pk = pk + 1
    return hessian

def function_CG_gradient(A, b, dk, epsilon=1e-6, k=0):
    '''
    Parameters
    ----------
    A : numpy.array
        矩阵
        
    b : numpy.array
        行向量
        
    dk : numpy.array
        初始梯度下降方向（列向量）
        
    epsilon : float
        精度
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        当前梯度（行向量）, 迭代次数
        
    '''
    import numpy as np
    rk = b.T - A.dot(dk)
    pk = rk
    while True:
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

def function_L_BFGS_double_loop(q, p, s, y, m, k, Hkm):
    '''
    Parameters
    ----------
    q : numpy.array
        初始梯度方向（行向量）
        
    p : list
        当前pk的列表
        
    s : list
        当前sk的列表
        
    y : list
        当前yk的列表
        
    m : int
        双循环阈值
        
    k : int
        迭代次数
        
    Hkm : numpy.array
        双循环初始矩阵
        

    Returns
    -------
    float
        当前梯度
        
    '''
    import numpy as np
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
    return - r.reshape(1, -1)

# 截断共轭梯度法实现
def function_Eq_Sovle(sk, pk, delta):
    '''
    Parameters
    ----------
    sk : float
        常数
        
    pk : float
        常数
        
    delta : float
        搜索半径
        

    Returns
    -------
    float
        大于0的方程解
        
    '''
    import sympy as sp
    m = sp.symbols("m", positive=True)
    r = (sk + m * pk)[0]
    sub = 0
    for i in r:
        sub += i**2
    h = sp.sqrt(sub) - delta
    mt = sp.solve(h)
    return mt[0]

def function_steihaug_CG(sk, rk, pk, B, delta, epsilon=1e-3, k=0):
    '''
    Parameters
    ----------
    s0 : list
        初始点列表
        
    rk : numpy.array
        梯度向量（行向量）
        
    pk : numpy.array
        负梯度向量（行向量）
        
    B : numpy.array
        修正后的海瑟矩阵
        
    delta : float
        搜索半径
        
    epsilon : float
        精度
        
    k : int
        迭代次数
        

    Returns
    -------
    float
        大于0的方程解
        
    '''
    import numpy as np
    s = []
    r = []
    p = []
    while True:
        s.append(sk)
        r.append(rk)
        p.append(pk)
        pbp = (p[k].dot(B)).dot(p[k].T)
        if pbp <= 0:
            m = function_Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        alphak = np.linalg.norm(r[k])**2 / pbp
        sk = s[k] + alphak * p[k]
        if np.linalg.norm(sk) > delta:
            m = function_Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        rk = r[k] + alphak * (B.dot(p[k].T)).T
        if np.linalg.norm(rk) < epsilon * np.linalg.norm(r[0]):
            ans = sk
            break
        betak = np.linalg.norm(rk)**2 / np.linalg.norm(r[k])**2
        pk = - rk + betak * p[k]
        k = k + 1
    return ans.astype(np.float64), k

def function_cons_unequal_L(cons_unequal, args, muk, sigma, x_0):
    '''
    Parameters
    ----------
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        当前不等式约束列表
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
    
    muk : list
        因子列表
        
    sigma : float
        常数
        
    x_0 : list or tuple
        当前迭代点列表（或元组）
        

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        加入因子约束后的不等式约束方程
        
    '''
    import numpy as np
    import sympy as sp
    sub = 0
    for i in range(cons_unequal.shape[0]):
        cons = muk[i] / sigma + cons_unequal[i]
        con = sp.Matrix([cons])
        conv = np.array(con.subs(dict(zip(args, x_0)))).astype(np.float64)
        if conv > 0:
            sub = sub + (cons**2 - (muk[i] / sigma)**2)
        else:
            sub = sub - (muk[i] / sigma)**2
    sub = sp.Matrix([sub])
    return sub

def function_v_k(cons_equal, cons_unequal, args, muk, sigma, x_0):
    '''
    Parameters
    ----------
    cons_equal : sympy.matrices.dense.MutableDenseMatrix
        当前等式约束列表
    
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        当前不等式约束列表
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
    
    muk : list
        因子列表
        
    sigma : float
        常数
        
    x_0 : list or tuple
        当前迭代点列表（或元组）
        

    Returns
    -------
    float
        终止常数
        
    '''
    import numpy as np
    sub = 0
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
    if cons_equal is not None:
        consv_equal = np.array(cons_equal.subs(reps)).astype(np.float64)
        sub += (consv_equal.T).dot(consv_equal)
        for i in range(len_unequal):
            sub += (max(consv_unequal[i], - muk[i] / sigma))**2
    else:
        for i in range(len_unequal):
            sub += (max(consv_unequal[i], - muk[i] / sigma))**2
    return np.sqrt(sub)

def function_renew_mu_k(cons_unequal, args, muk, sigma, x_0):
    '''
    Parameters
    ----------
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        当前不等式约束列表
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
    
    muk : list
        因子列表
        
    sigma : float
        常数
        
    x_0 : list or tuple
        当前迭代点列表（或元组）
        

    Returns
    -------
    list
        更新后的muk
        
    '''
    import numpy as np
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
    for i in range(len_unequal):
        muk[i] = max(muk[i] + sigma * consv_unequal[i], 0)
    return muk
