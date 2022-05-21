import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# 绘图函数
def plot_iteration(f, draw, method):
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
    if draw is True:
        plt.plot([i for i in range(len(f))], f, marker='o', c="teal", ls='--')
        plt.xlabel("$k$")
        plt.ylabel("$f(x_k)$")
        plt.title(method)
        plt.show()
    return None

# 取值
def f_x_k(funcs, args, x_0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程
        
    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list or tuple
        初始迭代点列表（或元组）
        

    Returns
    -------
    float
        迭代函数值
        
    '''
    funcsv = np.array(funcs.subs(dict(zip(args, x_0)))).astype(np.float64)
    return funcsv[0][0]

# 数据转换
def data_convert(funcs, args):
    '''
    Parameters
    ----------
    funcs : list or tuple or single value
        目标函数
        
    args : list or tuple or single value 
        参数


    Returns
    -------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        目标函数
        
    args : sympy.matrices.dense.MutableDenseMatrix 
        参数
        
    '''
    # convert funcs
    if funcs is not None:
        if isinstance(funcs, (list, tuple)):
            funcs = sp.Matrix(funcs)
        else:
            funcs = sp.Matrix([funcs])

    # convert args
    if args is not None:
        if isinstance(args, (list, tuple)):
            args = sp.Matrix(args)
        else:
            args = sp.Matrix([args])
    return funcs, args

# P_I(y, \sigma_k)
def neg_log(funcs, sigma, args, x_0, tk=0.02, epsilon=1e-10, k=0):
    '''
    Parameters
    ----------
    funcs : sympy.matrices.dense.MutableDenseMatrix
        当前目标方程

    sigma : float
    	罚项系数

    args : sympy.matrices.dense.MutableDenseMatrix
        参数列表
        
    x_0 : list
        初始迭代点列表

    tk : float
    	固定步长

    epsilon : double
        迭代停机准则
        
    k : int
        迭代次数
        

    Returns
    -------
    tuple
        最终收敛点, 迭代次数
        
    '''
    assert tk > 0
    funcs, args = data_convert(funcs, args)
    res = funcs.jacobian(args)
    point = []
    while 1:
        reps = dict(zip(args, x_0))
        point.append(x_0)
        grad = np.array(res.subs(reps)).astype(np.float64)
        x_0 = ((x_0 - tk * grad[0]) + np.sqrt((x_0 - tk * grad[0])**2 + 4 * tk * sigma)) / 2
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(x_0)
            break
    return x_0, k

# 求解主函数
def solve(funcs, args, x_0, draw=True, sigma=6, p=0.6, epsilon=1e-10, k=0):
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
    assert sigma > 0
    assert p > 0
    assert p < 1
    funcs, args = data_convert(funcs, args)
    point = []
    f = []
    while 1:
        point.append(np.array(x_0))
        f.append(f_x_k(funcs, args, x_0))
        x_0, _ = neg_log(funcs, sigma, args, tuple(x_0))
        k = k + 1
        sigma = p * sigma
        print(np.linalg.norm(x_0 - point[k - 1]))
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(f_x_k(funcs, args, x_0))
            break
    plot_iteration(f, draw, "proximity_operators_log")
    return x_0, k

# 测试用例1
x1, x2 = sp.symbols("x1 x2")
# f = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
# print(solve(f, [x1, x2], (2, 3)))

# 测试用例2
# f = (x2-1)**2 + (-x1+1)**2
# print(solve(f, [x1, x2], (2, 1)))

# 测试用例3
# f = (1-x1+x2)**2 + (1-x2)**2
# print(solve(f, [x1, x2], (2, 1)))

# 测试用例4
f = (1 - x1)**2 + 2*(1-x1)*(1-x2) + (1-x1)**2 + 2*(1-x1)-2*(1-x2)
print(solve(f, [x1, x2], (2, 3)))