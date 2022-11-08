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

__all__ = ['get_value', 'plot_iteration']

from ._typing import DataType, Optional, SympyMutableDenseMatrix, List, IterPointType, Union, PointType

def get_value(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, mu: Optional[float]=None) -> DataType:
    '''
    Parameters
    ----------
    funcs : SympyMutableDenseMatrix
        当前目标方程
        
    args : SympyMutableDenseMatrix
        参数列表
        
    x_0 : IterPointType
        初始迭代点
        
    mu : Optional[float]
        正则化参数
        

    Returns
    -------
    DataType
        迭代函数值
        
    '''
    import numpy as np
    funcsv = np.array(funcs.subs(dict(zip(args, x_0)))).astype(DataType)
    if mu is not None:
        for i in x_0:
            funcsv += mu * np.abs(i)
    return funcsv[0][0]

def plot_iteration(f: List[DataType], draw: bool, method: str) -> None:
    '''
    Parameters
    ----------
    f : List[DataType]]
        迭代函数值列表
        
    draw : bool
        绘图参数
        
    method : str
        最优化方法
        

    Returns
    -------
    None
        
    '''
    import matplotlib.pyplot as plt
    if draw is True:
        plt.plot([i for i in range(len(f))], f, marker='o', c="firebrick", ls='--')
        plt.xlabel("$k$")
        plt.ylabel("$f(x_k)$")
        plt.title(method)
        plt.show()
    return None