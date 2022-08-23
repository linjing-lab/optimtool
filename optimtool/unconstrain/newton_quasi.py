# BFGS拟牛顿法
def bfgs(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-10, k=0):
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
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : float
        海瑟矩阵条件数阈值
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.linear_search import armijo, goldstein, wolfe
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_modify_hessian, function_data_convert
    funcs, args, _, _ = function_data_convert(funcs, args)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(np.float64)
    hess = function_modify_hessian(hess, m)
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - np.linalg.inv(hess).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - np.array(res.subs(reps)).astype(np.float64)
            if yk.all != 0:
                hess = hess + (yk.T).dot(yk) / sk.dot(yk.T) - (hess.dot(sk.T)).dot((hess.dot(sk.T)).T) / sk.dot((hess.dot(sk.T)))
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_bfgs_" + method)
    return x_0, k, f if output_f is True else x_0, k

# DFP拟牛顿法
def dfp(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-4, k=0):
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
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : float
        海瑟矩阵条件数阈值
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.linear_search import armijo, goldstein, wolfe
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_modify_hessian, function_data_convert
    funcs, args, _, _ = function_data_convert(funcs, args)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(np.float64)
    hess = function_modify_hessian(hess, m)
    hessi = np.linalg.inv(hess)
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - hessi.dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - np.array(res.subs(reps)).astype(np.float64)
            if yk.all != 0:
                hessi = hessi - (hessi.dot(yk.T)).dot((hessi.dot(yk.T)).T) / yk.dot(hessi.dot(yk.T)) + (sk.T).dot(sk) / yk.dot(sk.T)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_dfp_" + method)
    return x_0, k, f if output_f is True else x_0, k

# L_BFGS方法
def L_BFGS(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=6, epsilon=1e-10, k=0):
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
        单调线搜索方法："armijo", "goldstein", "wolfe"
        
    m : double
        海瑟矩阵条件数阈值
        
    epsilon : double
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.linear_search import armijo, goldstein, wolfe
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_L_BFGS_double_loop, function_data_convert
    funcs, args, _, _ = function_data_convert(funcs, args)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    l = hes.shape[0]
    f = []
    s = []
    y = []
    p = []
    gamma = []
    gamma.append(1)
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        Hkm = gamma[k] * np.identity(l)
        grad = np.array(res.subs(reps)).astype(np.float64)
        dk = function_L_BFGS_double_loop(grad, p, s, y, m, k, Hkm)
        if np.linalg.norm(dk) >= epsilon:
            alphak = eval(method)(funcs, args, x_0, dk)
            x_0 = x_0 + alphak * dk[0]
            if k > m:
                s[k - m] = np.empty((1, l))
                y[k - m] = np.empty((1, l))
            sk = alphak * dk
            s.append(sk)
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - grad
            y.append(yk)
            pk = 1 / yk.dot(sk.T)
            p.append(pk)
            gammak = sk.dot(sk.T) / yk.dot(yk.T)
            gamma.append(gammak)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_L_BFGS_" + method)
    return x_0, k, f if output_f is True else x_0, k