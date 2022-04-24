# 高斯-牛顿法（非线性最小二乘问题）
def gauss_newton(funcr, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcr : sympy.matrices.dense.MutableDenseMatrix
        当前目标残差方程
        
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
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import sympy as sp
    import numpy as np
    from optimtool.functions.linear_search import armijo, goldstein, wolfe
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(np.float64)
        f.append(function_f_x_k(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(np.float64)
        q, r = np.linalg.qr(jk)
        dk = np.linalg.inv(r).dot(-(q.T).dot(rk)).reshape(1,-1)
        if np.linalg.norm(dk) > epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "nonlinear_least_square_gauss_newton_" + method)
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# levenberg marquardt方法
def levenberg_marquardt(funcr, args, x_0, draw=True, output_f=False, m=100, lamk=1, eta=0.2, p1=0.4, p2=0.9, gamma1=0.7, gamma2=1.3, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcr : sympy.matrices.dense.MutableDenseMatrix
        当前目标残差方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    m : float
        海瑟矩阵条件数阈值
        
    lamk : int
        修正常数
        
    eta : float
        常数
        
    p1 : float 
        常数
        
    p2 : float
        常数
        
    gamma1 : float
        常数
        
    gamma2 : float
        常数
        
    epsilon : float
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import sympy as sp
    import numpy as np
    from optimtool.functions.linear_search import armijo, goldstein, wolfe
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_modify_hessian, function_CG_gradient
    assert eta >= 0
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    resf = funcs.jacobian(args)
    hess = resf.jacobian(args)
    dk0 = np.zeros((args.shape[0], 1))
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(np.float64)
        f.append(function_f_x_k(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(np.float64)
        dk, _ = function_CG_gradient((jk.T).dot(jk) + lamk, -((jk.T).dot(rk)).reshape(1, -1), dk0)
        pk_up = np.array(funcs.subs(reps)).astype(np.float64) - np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(np.float64)
        grad_f = np.array(resf.subs(reps)).astype(np.float64)
        hess_f = np.array(hess.subs(reps)).astype(np.float64)
        hess_f = function_modify_hessian(hess_f, m)
        pk_down = - (grad_f.dot(dk.T) + 0.5*((dk.dot(hess_f)).dot(dk.T)))
        pk = pk_up / pk_down
        if np.linalg.norm(dk) >= epsilon:
            if pk < p1:
                lamk = gamma2 * lamk
            else:
                if pk > p2:
                    lamk = gamma1 * lamk
                else:
                    lamk = lamk
            if pk > eta:
                x_0 = x_0 + dk[0]
            else:
                x_0 = x_0
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "nonlinear_least_square_levenberg_marquardt")        
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
