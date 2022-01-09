'''
牛顿类算法
'''
# 经典牛顿法
def classic(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0):
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
        非单调线搜索方法："grippo"与"ZhangHanger"
        
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
    from functions.tools import function_f_x_k, function_plot_iteration
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        hessian = np.array(hes.subs(reps)).astype(np.float64)
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - np.linalg.inv(hessian).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            x_0 = x_0 + dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_classic")        
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
    
# 修正牛顿法
def modified(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-10, k=0):
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
    from functions.tools import function_f_x_k, function_plot_iteration, function_modify_hessian
    from functions.linear_search import armijo, goldstein, wolfe
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        hess = np.array(hes.subs(reps)).astype(np.float64)
        hessian = function_modify_hessian(hess, m)
        dk = - np.linalg.inv(hessian).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_modified_" + method)
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# 非精确牛顿法
def CG(funcs, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-6, k=0):
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
    from functions.linear_search import armijo, goldstein, wolfe
    from functions.tools import function_f_x_k, function_plot_iteration, function_CG_gradient
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    dk0 = np.zeros((args.shape[0], 1))
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        hess = np.array(hes.subs(reps)).astype(np.float64)
        # 采用共轭梯度法求解梯度
        dk, _ = function_CG_gradient(hess, - gradient, dk0)
        if np.linalg.norm(dk) >= epsilon:
            alpha = eval(method)(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_CG_" + method)
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k