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

from .unconstrain.gradient_descent import barzilar_borwein
from .unconstrain.newton import CG
from .unconstrain.newton_quasi import L_BFGS
from .unconstrain.trust_region import steihaug_CG

def kernel(method: str) -> str:
    '''
    method : str
        无约束优化方法内核


    Returns
    -------
    str
        内核方法名
        
    '''
    if method == "gradient_descent":
        return 'barzilar_borwein'
    elif method == "newton":
        return 'CG'
    elif method == "newton_quasi":
        return 'L_BFGS'
    elif method == "trust_region":
        return 'steihaug_CG'
    return 'barzilar_borwein'