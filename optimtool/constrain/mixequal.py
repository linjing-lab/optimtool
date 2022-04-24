# 二次罚函数法（混合约束）
def penalty_quadratic(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", sigma=10, p=0.6, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons_equal : sympy.matrices.dense.MutableDenseMatrix
        等式参数约束列表
        
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
        
    sigma : float
        罚函数因子
        
    p : float
        修正参数
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 0
    # data convert
    if isinstance(args, list) or isinstance(args, tuple):
        funcs = sp.Matrix(funcs)
    else:
        funcs = sp.Matrix([funcs])
    if isinstance(args, list) or isinstance(args, tuple):
        args = sp.Matrix(args)
    else:
        args = sp.Matrix([args])
    if isinstance(cons_equal, list) or isinstance(cons_equal, tuple):
        cons_equal = sp.Matrix(cons_equal)
    else:
        cons_equal = sp.Matrix([cons_equal])
    if isinstance(cons_unequal, list) or isinstance(cons_unequal, tuple):
        cons_unequal = sp.Matrix(cons_unequal)
    else:
        cons_unequal = sp.Matrix([cons_unequal])
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons_unequal.subs(reps)).astype(np.float64)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons_unequal.T * consv + (sigma / 2) * cons_equal.T * cons_equal])
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
    function_plot_iteration(f, draw, "penalty_quadratic_mixequal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# 精确罚函数法-l1罚函数法（混合约束）
def penalty_L1(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", sigma=1, p=0.6, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons_equal : sympy.matrices.dense.MutableDenseMatrix
        等式参数约束列表
        
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
        
    sigma : float
        罚函数因子
        
    p : float
        修正参数
        
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
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 0
    # data convert
    if isinstance(args, list) or isinstance(args, tuple):
        funcs = sp.Matrix(funcs)
    else:
        funcs = sp.Matrix([funcs])
    if isinstance(args, list) or isinstance(args, tuple):
        args = sp.Matrix(args)
    else:
        args = sp.Matrix([args])
    if isinstance(cons_equal, list) or isinstance(cons_equal, tuple):
        cons_equal = sp.Matrix(cons_equal)
    else:
        cons_equal = sp.Matrix([cons_equal])
    if isinstance(cons_unequal, list) or isinstance(cons_unequal, tuple):
        cons_unequal = sp.Matrix(cons_unequal)
    else:
        cons_unequal = sp.Matrix([cons_unequal])
    point = []
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
        consv_unequal = np.where(consv_unequal <= 0, consv_unequal, 1)
        consv_unequal = np.where(consv_unequal > 0, consv_unequal, 0)
        consv_equal = np.array(cons_equal.subs(reps)).astype(np.float64)
        consv_equal = np.where(consv_equal <= 0, consv_equal, 1)
        consv_equal = np.where(consv_equal > 0, consv_equal, -1)
        pe = sp.Matrix([funcs + sigma * cons_unequal.T * consv_unequal + sigma * cons_equal.T * consv_equal])
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
    function_plot_iteration(f, draw, "penalty_L1")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k

# 增广拉格朗日函数法（混合约束）
def lagrange_augmented(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", lamk=6, muk=10, sigma=8, alpha=0.5, beta=0.7, p=2, eta=1e-3, epsilon=1e-4, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    cons_equal : sympy.matrices.dense.MutableDenseMatrix
        等式参数约束列表
        
    cons_unequal : sympy.matrices.dense.MutableDenseMatrix
        不等式参数约束列表
        
    x_0 : list
        初始迭代点列表
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
    method : string
        无约束优化方法内核
        
    lamk : float
        因子
    
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
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_f_x_k, function_plot_iteration, function_cons_unequal_L, function_v_k, function_renew_mu_k
    from optimtool.unconstrain.gradient_descent import barzilar_borwein
    from optimtool.unconstrain.newton import CG
    from optimtool.unconstrain.newton_quasi import L_BFGS
    from optimtool.unconstrain.trust_region import steihaug_CG
    assert sigma > 0
    assert p > 1
    assert alpha > 0 
    assert alpha <= beta
    assert beta < 1
    # data convert
    if isinstance(args, list) or isinstance(args, tuple):
        funcs = sp.Matrix(funcs)
    else:
        funcs = sp.Matrix([funcs])
    if isinstance(args, list) or isinstance(args, tuple):
        args = sp.Matrix(args)
    else:
        args = sp.Matrix([args])
    if isinstance(cons_equal, list) or isinstance(cons_equal, tuple):
        cons_equal = sp.Matrix(cons_equal)
    else:
        cons_equal = sp.Matrix([cons_equal])
    if isinstance(cons_unequal, list) or isinstance(cons_unequal, tuple):
        cons_unequal = sp.Matrix(cons_unequal)
    else:
        cons_unequal = sp.Matrix([cons_unequal])
    f = []
    lamk = np.array([lamk for i in range(cons_equal.shape[0])]).reshape(cons_equal.shape[0], 1)
    muk = np.array([muk for i in range(cons_unequal.shape[0])]).reshape(cons_unequal.shape[0], 1)
    while 1:
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = function_cons_unequal_L(cons_unequal, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * (cons_equal.T * cons_equal + cons_uneuqal_modifyed) + cons_equal.T * lamk])
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
        vkx = function_v_k(cons_equal, cons_unequal, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64)) <= eta):
                f.append(function_f_x_k(funcs, args, x_0))
                break
            else:
                lamk = lamk + sigma * np.array(cons_equal.subs(dict(zip(args, x_0)))).astype(np.float64)
                muk = function_renew_mu_k(cons_unequal, args, muk, sigma, x_0)
                sigma = sigma
                etak = etak / sigma
                epsilonk = epsilonk / sigma**beta
        else:
            lamk = lamk
            sigma = p * sigma
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    function_plot_iteration(f, draw, "lagrange_augmented_mixequal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k