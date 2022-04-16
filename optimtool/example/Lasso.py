def gradient_descent(A, b, mu, args, x_0, draw=True, output_f=False, delta=10, alp=1e-3, epsilon=1e-2, k=0):
    '''
    Parameters
    ----------
    A : numpy.array
        m*n维数 参数矩阵
        
    b : 系数矩阵
        m*1维数 参数矩阵
        
    mu : float
        正则化参数
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    delta : float
        常数
        
    alp : float
        步长阈值
        
    epsilon : float
        迭代停机准则
        
    k : float
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_get_f_delta_gradient
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    L = np.linalg.norm((A.T).dot(A)) + mu / delta
    point = []
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0, mu))
        resv = np.array(res.subs(reps)).astype(np.float64)
        argsv = np.array(args.subs(reps)).astype(np.float64)
        g = function_get_f_delta_gradient(resv, argsv, mu, delta)
        alpha = alp
        assert alpha <= 1 / L
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0, mu))
            break
    function_plot_iteration(f, draw, "Lasso_gradient_decent")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

'''
次梯度算法
'''
def subgradient(A, b, mu, args, x_0, draw=True, output_f=False, alphak=2e-2, epsilon=1e-3, k=0):
    '''
    Parameters
    ----------
    A : numpy.array
        m*n维数 参数矩阵
        
    b : 系数矩阵
        m*1维数 参数矩阵
        
    mu : float
        正则化参数
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    alphak : float
        自适应步长参数
        
    epsilon : float
        迭代停机准则
        
    k : float
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_get_subgradient
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    point = []
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0, mu))
        resv = np.array(res.subs(reps)).astype(np.float64)
        argsv = np.array(args.subs(reps)).astype(np.float64)
        g = function_get_subgradient(resv, argsv, mu)
        alpha = alphak / np.sqrt(k + 1)
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0, mu))
            break
    function_plot_iteration(f, draw, "Lasso_subgradient")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k