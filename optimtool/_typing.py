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

from .base import np, sp

from typing import (
    Tuple,
    List,
    Union,
    Optional
)

NDArray = np.ndarray # numpy.ndarray
SympyMutableDenseMatrix = sp.matrices.dense.MutableDenseMatrix # symbolic values

DataType = np.float64 # conver `object` to `float`
# add type of FuncType to support need!
ArgType = sp.core.symbol.Symbol
MulType = sp.core.mul.Mul
PowerType = sp.core.power.Pow
AddType = sp.core.add.Add 
FuncType = Union[AddType, PowerType, MulType, ArgType]
PointType = Union[float, int]

FuncArray = Union[SympyMutableDenseMatrix, FuncType, List[FuncType], Tuple[FuncType]]
ArgArray = Union[SympyMutableDenseMatrix, ArgType, List[ArgType], Tuple[ArgType]]
PointArray = Union[PointType, List[PointType], Tuple[PointType]]

IterPointType = Union[NDArray, List[PointType], Tuple[PointType]] # support more situations

OutputType = Union[Tuple[IterPointType, int], Tuple[IterPointType, int, List[DataType]]]