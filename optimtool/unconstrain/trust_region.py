# Copyright (c) 2021 linjing-lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from .._utils import plot_iteration
from .._convert import f2m, a2m, p2t, h2h

from .._typing import FuncArray, ArgArray, PointArray, OutputType, DataType

# 信赖域算法
def steihaug_CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, draw: bool=True, output_f: bool=False, r0: float=1, rmax: float=2, eta: float=0.2, p1: float=0.4, p2: float=0.6, gamma1: float=0.5, gamma2: float=1.5, epsilon: float=1e-6, k: int=0) -> OutputType:
    '''
    Parameters
    ----------
    funcs : FuncArray
        当前目标方程
        
    args : ArgArray
        参数
        
    x_0 : PointArray
        初始迭代点
        
    draw : bool
        绘图接口参数
        
    output_f : bool
        输出迭代函数值列表
        
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
        
    k : int
        迭代次数
        

    Returns
    -------
    OutputType
        最终收敛点, 迭代次数, (迭代函数值列表)
        
    '''
    assert eta >= 0
    assert r0 < rmax
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    from .._drive import steihaug
    funcs, args, x_0 = f2m(funcs), a2m(args), p2t(x_0)
    assert all(funcs.shape) == 1 and args.shape[0] == len(x_0)
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    s0, f = [0 for _ in range(args.shape[0])], []
    while 1:
        reps = dict(zip(args, x_0))
        funv = np.array(funcs.subs(reps)).astype(DataType)
        f.append(funv[0][0])
        grad = np.array(res.subs(reps)).astype(DataType)
        hessi = np.array(hes.subs(reps)).astype(DataType)
        hessi = h2h(hessi)
        dk, _ = steihaug(s0, grad, -grad, hessi, r0)
        if np.linalg.norm(dk) >= epsilon:
            funvk = np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(DataType)
            pk = (funv - funvk) / -(grad.dot(dk.T) + 0.5*((dk.dot(hessi)).dot(dk.T)))
            if pk < p1:
                r0 = gamma1 * r0
            else:
                if (pk > p2) or (np.linalg.norm(dk) == r0):
                    r0 = min(gamma2 * r0, rmax)
            if pk > eta:
                x_0 = x_0 + dk[0]
            k += 1
        else:
            break
    plot_iteration(f, draw, "trust_region_steihaug_CG")
    return (x_0, k, f) if output_f is True else (x_0, k)

__all__ = [steihaug_CG]