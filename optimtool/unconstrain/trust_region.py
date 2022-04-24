# 信赖域算法
def steihaug_CG(funcs, args, x_0, draw=True, output_f=False, m=100, r0=1, rmax=2, eta=0.2, p1=0.4, p2=0.6, gamma1=0.5, gamma2=1.5, epsilon=1e-6, k=0):
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
        
    m : float
        海瑟矩阵条件数阈值
        
    r0 : float
        搜索半径起点
        
    rmax : float
        搜索最大半径
        
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
        
    k : float
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    import numpy as np
    import sympy as sp
    from optimtool.functions.tools import function_modify_hessian, function_steihaug_CG, function_plot_iteration
    assert eta >= 0
    assert r0 < rmax
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    # data convert
    if isinstance(funcs, list) or isinstance(funcs, tuple):
        funcs = sp.Matrix(funcs)
    else:
        funcs = sp.Matrix([funcs])
    if isinstance(args, list) or isinstance(args, tuple):
        args = sp.Matrix(args)
    else:
        args = sp.Matrix([args])
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    s0 = [0 for i in range(args.shape[0])]
    f = []
    while 1:
        reps = dict(zip(args, x_0))
        funv = np.array(funcs.subs(reps)).astype(np.float64)
        f.append(funv[0][0])
        grad = np.array(res.subs(reps)).astype(np.float64)
        hessi = np.array(hes.subs(reps)).astype(np.float64)
        hessi = function_modify_hessian(hessi, m)
        dk, _ = function_steihaug_CG(s0, grad, - grad, hessi, r0)
        if np.linalg.norm(dk) >= epsilon:
            funvk = np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(np.float64)
            pk = (funv - funvk) / -(grad.dot(dk.T) + 0.5*((dk.dot(hessi)).dot(dk.T)))
            if pk < p1:
                r0 = gamma1 * r0
            else:
                if (pk > p2) | (np.linalg.norm(dk) == r0):
                    r0 = min(gamma2 * r0, rmax)
                else:
                    r0 = r0
            if pk > eta:
                x_0 = x_0 + dk[0]
            else:
                x_0 = x_0
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "trust_region_steihaug_CG")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k