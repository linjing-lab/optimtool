# h(x)=||x||_1
def L1(funcs, mu, gfun, args, x_0, draw=True, output_f=False, t=0.01, epsilon=1e-6, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
    
    mu : float
        l1范数前系数

	gfun : sympy.matrices.dense.MutableDenseMatrix
		与args参数个数相同

    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表

    t : float
        学习率
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert, function_proximity_L1
    assert t > 0
    funcs, args, gfun, _ = function_data_convert(funcs, args, gfun)
    res = funcs.jacobian(args)
    f = []
    point = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0, mu))
        point.append(x_0)
        grad = np.array(res.subs(reps)).astype(np.float64)
        x_0 = function_proximity_L1(mu, gfun, args, x_0, grad, t)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            f.append(function_f_x_k(funcs, args, x_0, mu))
            break
    function_plot_iteration(f, draw, "approximate_point_gradient_L1")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# h(x)=\sum_{i=1}^{n} ln x_i
def neg_log(funcs, mu, gfun, args, x_0, draw=True, output_f=False, t=0.01, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
    
    mu : float
        负对数前系数

    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表

    t : float
        学习率
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert, function_proximity_neg_log
    assert t > 0
    funcs, args, gfun, _ = function_data_convert(funcs, args, gfun)
    res = funcs.jacobian(args)
    f = []
    point = []
    while 1:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0, mu))
        point.append(x_0)
        grad = np.array(res.subs(reps)).astype(np.float64)
        x_0 = function_proximity_neg_log(mu, gfun, args, x_0, grad, t)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            f.append(function_f_x_k(funcs, args, x_0, mu))
            break
    function_plot_iteration(f, draw, "approximate_point_gradient_neg_log")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k