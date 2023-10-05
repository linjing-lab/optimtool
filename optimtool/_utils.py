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

from .base import np, plt
from ._typing import DataType, Optional, SympyMutableDenseMatrix, List, IterPointType

__all__ = ["get_value", "plot_iteration"]

def get_value(funcs: SympyMutableDenseMatrix, args: SympyMutableDenseMatrix, x_0: IterPointType, mu: Optional[float]=None, proxim: Optional[str]=None) -> DataType:
    '''
    :param funcs: SympyMutableDenseMatrix, objective function after `convert`.
    :param args: SympyMutableDenseMatrix, symbolic set after `convert` with order.
    :param x_0: IterPointType, numpy.ndarray or List[PointType] or Tuple[PointType].
    :param mu: float | None, parameters collaborate with the problems applied in `Lasso`. default=None.

    :return: functional value with DataType.
    '''
    funcsv = np.array(funcs.subs(dict(zip(args, x_0)))).astype(DataType)
    if mu is not None:
        if proxim == "L1":
            funcsv += mu * np.sum(np.abs(x_0))
        elif proxim == "L2":
            funcsv += mu * np.linalg.norm(x_0)
        elif proxim == "ln":
            funcsv += -mu * np.sum(np.log(x_0))
    return funcsv[0][0]

def plot_iteration(f: List[DataType], draw: bool, method: str) -> None:
    '''
    :param f: List[DataType], iterative value with format `DataType` in a list.
    :param draw: bool, use `bool` to control whether to draw visual images.
    :param method: method appearing on the visual interface.
    '''
    if draw is True:
        plt.plot([i for i in range(len(f))], f, marker='o', c="firebrick", ls='--')
        plt.xlabel("$k$")
        plt.ylabel("$f(x_k)$")
        plt.title(method)
        plt.show()
    return None