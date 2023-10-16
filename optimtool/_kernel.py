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
    :param method: str, unconstrained methods serving as the kernel of constraint methods.

    :return: executable unconstrained kernels compatible with constrained porcess.
    ''' 
    from .unconstrain.gradient_descent import barzilar_borwein
    from .unconstrain.newton import CG
    from .unconstrain.newton_quasi import bfgs
    from .unconstrain.trust_region import steihaug_CG
    # set the above methods by change name after `import`, like bfgs -> L_BFGS.
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
    :param method: str, linear search methods serving as the alpha searcher of unconstrained method.

    :return: executable linear search functions compatible with subsequent alpha search operations.
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
    :param method: str, non-monotonic linear search serving as the alpha searcher of barzilar_borwein.
    :param M: int, constant used to control the inner `max` process of `Grippo`.
    :param eta: float, constant used to control `C_k` process of `ZhangHanger`.

    :return: executable functions compatible with barzilar_borwein alpha search operations.
    '''
    from ._search import Grippo, ZhangHanger
    if method == 'Grippo':
        return Grippo, M
    elif method == 'ZhangHanger':
        return ZhangHanger, eta
    else:
        raise ValueError("The search selector supports 2 parameters: Grippo, ZhangHanger.")

def set_proxim(method: str):
    '''
    :param method: str, the different proximity operators used in updating iteration point noted with `method`.

    :return: executable functions compatible with updating iteration point of hybrid algorithms. 
    '''
    from ._proxim import l1, l2, ln
    if method == 'L1':
        return l1
    elif method == 'L2':
        return l2
    elif method == 'ln':
        return ln
    else:
        raise ValueError("The proximity operators selector supports 3 parameters: L1, L2, ln.")