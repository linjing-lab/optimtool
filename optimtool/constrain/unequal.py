# 二次罚函数法（不等式约束）
def penalty_quadratic(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", sigma=10, p=0.4, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 0
    assert p < 1
    funcs, args, _, cons = function_data_convert(funcs, args, None, cons)
    point = []
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(np.float64)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons.T * consv])
        if method == "gradient_descent":
            x_0, _ = barzilar_borwein(pe, args, tuple(x_0), draw=False)
        elif method == "newton":
            x_0, _ = CG(pe, args, tuple(x_0), draw=False)
        elif method == "newton_quasi":
            x_0, _ = L_BFGS(pe, args, tuple(x_0), draw=False)
        elif method == "trust_region":
            x_0, _ = steihaug_CG(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_unequal") 
    return x_0, k, f if output_f is True else x_0, k

# 内点罚函数法（不等式约束）
'''
保证点在定义域内
'''
def penalty_interior_log(funcs, args, cons, x_0, draw=True, output_f=False, sigma=12, p=0.6, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert
    from optimtool.hybrid.approximate_point_gradient import neg_log
    assert sigma > 0
    assert p > 0
    assert p < 1
    funcs, args, _, cons = function_data_convert(funcs, args, None, cons)
    point = []
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        x_0, _ = neg_log(funcs, sigma, -cons, args, tuple(x_0), draw=False)
        k = k + 1
        sigma = p * sigma
        print(np.linalg.norm(x_0 - point[k - 1]))
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
    function_plot_iteration(f, draw, "penalty_interior_fraction")
    return x_0, k, f if output_f is True else x_0, k

# 分式
def penalty_interior_fraction(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", sigma=12, p=0.6, epsilon=1e-6, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
        
    sigma : double
        罚函数因子
        
    p : double
        修正参数
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_data_convert
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 0
    assert p < 1
    funcs, args, _, cons = function_data_convert(funcs, args, None, cons)
    point = []
    f = []
    sub_pe = 0
    for i in cons:
        sub_pe += 1 / i
    sub_pe = sp.Matrix([sub_pe])
    while 1:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = sp.Matrix([funcs - sigma * sub_pe])
        if method == "gradient_descent":
            x_0, _ = barzilar_borwein(pe, args, tuple(x_0), draw=False)
        elif method == "newton":
            x_0, _ = CG(pe, args, tuple(x_0), draw=False)
        elif method == "newton_quasi":
            x_0, _ = L_BFGS(pe, args, tuple(x_0), draw=False)
        elif method == "trust_region":
            x_0, _ = steihaug_CG(pe, args, tuple(x_0), draw=False)
        k = k + 1
        sigma = p * sigma
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, args, x_0))
            break
    function_plot_iteration(f, draw, "penalty_interior_fraction")
    return x_0, k, f if output_f is True else x_0, k
    
# 增广拉格朗日函数法（不等式约束）
def lagrange_augmented(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", muk=10, sigma=8, alpha=0.2, beta=0.7, p=2, eta=1e-1, epsilon=1e-4, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
    
    muk : float
        因子
    
    sigma : float
        罚函数因子
    
    alpha : float
        初始步长
    
    beta : float
        修正参数
    
    p : float
        修正参数
    
    eta : float
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_cons_unequal_L, function_renew_mu_k, function_v_k, function_data_convert
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 1
    assert alpha > 0 
    assert alpha <= beta
    assert beta < 1
    funcs, args, _, cons = function_data_convert(funcs, args, None, cons)
    f = []
    muk = np.array([muk for i in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while 1:
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = function_cons_unequal_L(cons, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * cons_uneuqal_modifyed])
        f.append(function_f_x_k(funcs, args, x_0))
        if method == "gradient_descent":
            x_0, _ = barzilar_borwein(L, args, x_0, draw=False, epsilon=etak)
        elif method == "newton":
            x_0, _ = CG(L, args, x_0, draw=False, epsilon=etak)
        elif method == "newton_quasi":
            x_0, _ = L_BFGS(L, args, x_0, draw=False, epsilon=etak)
        elif method == "trust_region":
            x_0, _ = steihaug_CG(L, args, x_0, draw=False, epsilon=etak)
        k = k + 1
        vkx = function_v_k(None, cons, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64)) <= eta):
                f.append(function_f_x_k(funcs, args, x_0))
                break
            else:
                muk = function_renew_mu_k(cons, args, muk, sigma, x_0)
                sigma = sigma
                etak = etak / sigma
                epsilonk = epsilonk / sigma**beta
        else:
            sigma = p * sigma
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    function_plot_iteration(f, draw, "lagrange_augmented_unequal")
    return x_0, k, f if output_f is True else x_0, k