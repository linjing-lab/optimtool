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

def kernel(method: str):
    '''
    Parameters
    ----------
    method : str
        无约束优化方法内核

        
    Returns
    -------
    .unconstrain.
        内核方法名
        
    '''
    from .unconstrain.gradient_descent import barzilar_borwein
    from .unconstrain.newton import CG
    from .unconstrain.newton_quasi import bfgs
    from .unconstrain.trust_region import steihaug_CG
    if method == "newton":
        return CG
    elif method == "gradient_descent":
        return barzilar_borwein
    elif method == "newton_quasi":
        return bfgs
    elif method == "trust_region":
        return steihaug_CG
    else:
        raise ValueError("The kernel selector supports 4 parameters: gradient_descent, newton, newton_quasi, trust_region.")
    
def linear_search(method: str):
    '''
    Parameters
    ----------
    method: str
        线搜索方法作为无约束方法的步长搜索器

        
    Returns
    -------
    ._search.
        与后续操作兼容的线搜索方法
    '''
    from ._search import armijo, goldstein, wolfe
    if method == 'armijo':
        return armijo
    elif method == 'goldstein':
        return goldstein
    elif method == 'wolfe':
        return wolfe
    else:
        raise ValueError("The search selector supports 3 parameters: armijo, goldstein, wolfe.")
    
def nonmonotonic_search(method: str, M: int, eta: float):
    '''
    Parameters
    ----------
    method: str
        非单调线搜索方法作为barzilar_borwein的步长搜索器
    
    M: int
        约束内部`max`过程的常量
    
    eta: float
        控制`C_k`过程的常量
        

    Returns
    -------
    (Grippo or ZhangHanger, int or float)
        与barzilar_borwein兼容的函数
    '''
    from ._search import Grippo, ZhangHanger
    if method == 'Grippo':
        return Grippo, M
    elif method == 'ZhangHanger':
        return ZhangHanger, eta
    else:
        raise ValueError("The search selector supports 2 parameters: Grippo, ZhangHanger.")