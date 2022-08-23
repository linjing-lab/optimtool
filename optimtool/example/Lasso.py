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
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_get_f_delta_gradient, function_data_convert
    _, args, _, _ = function_data_convert(None, args)
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
    return x_0, k, f if output_f is True else x_0, k

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
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_get_subgradient, function_data_convert
    _, args, _, _ = function_data_convert(None, args)
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
    return x_0, k, f if output_f is True else x_0, k

'''
罚函数法
'''
def penalty(A, b, mu, args, x_0, draw=True, output_f=False, gamma=0.1, epsilon=1e-6, k=0):
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
        
    gamma : float
        因子
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert
    assert gamma < 1
    assert gamma > 0
    _, args, _, _ = function_data_convert(None, args)
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    f = []
    while mu >= epsilon:
        f.append(function_f_x_k(funcs, args, x_0, mu))
        x_0, _ = subgradient(A, b, mu, args, x_0, False)
        if mu > epsilon:
            mu = max(epsilon, gamma * mu)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "Lasso_penalty")
    return x_0, k, f if output_f is True else x_0, k

'''
近似点梯度法
'''
def approximate_point_gradient(A, b, mu, args, x_0, draw=True, output_f=False, epsilon=1e-6, k=0):
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert
    _, args, _, _ = function_data_convert(None, args)
    values, _ = np.linalg.eig((A.T).dot(A))
    lambda_ma = max(values)
    if isinstance(lambda_ma, complex):
        tk = 1 / np.real(lambda_ma)
    else:
        tk = 1 / lambda_ma
    funcs = sp.Matrix([0.5*((A*args - b).T)*(A*args - b)])
    res = funcs.jacobian(args)
    f = []
    point = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0, mu))
        point.append(x_0)
        grad = np.array(res.subs(reps)).astype(np.float64)
        x_0 = np.sign(x_0 - tk * grad[0]) * [max(i, 0) for i in np.abs(x_0 - tk * grad[0]) - tk * mu]
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            f.append(function_f_x_k(funcs, args, x_0, mu))
            break
    function_plot_iteration(f, draw, "Lasso_approximate_point_gradient")
    return x_0, k, f if output_f is True else x_0, k