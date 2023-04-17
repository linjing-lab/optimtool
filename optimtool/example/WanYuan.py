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

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from .._typing import FuncType, Tuple, FuncArray, ArgArray

def solution(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float, x_0: tuple, draw: bool=False, eps: float=1e-10) -> None:
    '''
    Parameters
    ----------
    m : float
        直线的斜率
        
    n : float
        直线的截距
        
    a : float
        抛物线的二次项系数
        
    b : float
        抛物线的一次项系数
        
    c : float
        抛物线的常数项
        
    x3 : float
        所求圆须通过的定点的横坐标
        
    y3 : float
        所求圆须通过的定点的纵坐标
        
    x_0 : tuple
        初始点：(x0, y0, x1, y2, x2, y2)
        
    draw : bool
        绘图接口
        

    Returns
    -------
    None
        
    '''
    # 导入非线性最小二乘的包
    from ..unconstrain.nonlinear_least_square import gauss_newton
    # 构造残差函数
    def maker_line_1(m: float, n: float) -> FuncType:
        '''
        Parameters
        ----------
        m : float
            直线的斜率
            
        n : float
            直线的截距

        Returns
        -------
        方程
            (x1, y1)在直线上

        '''
        x1, y1 = sp.symbols("x1 y1")
        return m*x1 + n - y1

    def maker_line_2(x3: float, y3: float) -> FuncType:
        '''
        Parameters
        ----------
        x3 : float
            所求圆须通过的定点的横坐标
            
        y3 : float
            所求圆须通过的定点的纵坐标

        Returns
        -------
        方程
            (x1, y1) 点在圆上

        '''
        x1, y1, x0, y0 = sp.symbols("x1 y1 x0 y0")
        return (x1 - x0)**2 + (y1 - y0)**2 - ((x3 - x0)**2 + (y3 - y0)**2)

    def maker_line_3(m: float, n: float, x3: float, y3: float) -> FuncType:
        '''
        Parameters
        ----------
        m : float
            直线的斜率
            
        n : float
            直线的截距
            
        x3 : float
            所求圆须通过的定点的横坐标
            
        y3 : float
            所求圆须通过的定点的纵坐标

        Returns
        -------
        方程
            圆与直线相切: 判别式为0

        '''
        x, y, x0, y0= sp.symbols("x y x0 y0")
        delta = (2*m*n - 2*x - 2*m*y)**2 - 4*(m**2 + 1)*(x**2 + y**2 - (x3 - x)**2 + (y3 - y)**2 + n**2 - 2*n*y)
        return delta.subs({x: x0, y: y0})

    def maker_line_4(m: float) -> FuncType:
        '''
        Parameters
        ----------
        m : float
            直线的斜率

        Returns
        -------
        方程
            圆与直线在点(x1, y1)处斜率相等

        '''
        x1, y1, x0, y0= sp.symbols("x1 y1 x0 y0")
        return m*y1 - m*y0 + x1 - x0

    def maker_parabola_1(a: float, b: float) -> FuncType:
        '''
        Parameters
        ----------
        a : float
            抛物线的二次项系数
            
        b : float
            抛物线的一次项系数

        Returns
        -------
        eq : 方程
            抛物线与圆切线斜率相等

        '''
        x2, y2, x0, y0 = sp.symbols("x2 y2 x0 y0")
        eq = 2*a*x2*y2 - 2*a*x2*y0 + b*y2 - b*y0 - x0 + x2
        return eq

    def maker_parabola_2(a: float, b: float, c: float) -> FuncType:
        '''
        Parameters
        ----------
        a : float
            抛物线的二次项系数
            
        b : float
            抛物线的一次项系数
            
        c : float
            抛物线的常数项

        Returns
        -------
        方程
            (x2, y2)点在抛物线上

        '''
        x2, y2 = sp.symbols("x2 y2")
        return a*x2**2 + b*x2 + c - y2

    def maker_parabola_3(x3: float, y3: float) -> FuncType:
        '''
        Parameters
        ----------
        x3 : float
            所求圆须通过的定点的横坐标
            
        y3 : float
            所求圆须通过的定点的纵坐标

        Returns
        -------
        方程
            (x2, y2)点在圆上
            
        '''
        x2, y2, x0, y0 = sp.symbols("x2 y2 x0 y0")
        return (x2 - x0)**2 + (y2 - y0)**2 - ((x3 - x0)**2 + (y3 - y0)**2)

    def maker_data(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float) -> Tuple[FuncArray, ArgArray]:
        '''
        Parameters
        ----------
        m : float
            直线的斜率
            
        n : float
            直线的截距
            
        a : float
            抛物线的二次项系数
            
        b : float
            抛物线的一次项系数
            
        c : float
            抛物线的常数项
            
        x3 : float
            所求圆须通过的定点的横坐标
            
        y3 : float
            所求圆须通过的定点的纵坐标
            

        Returns
        -------
        tuple :
            残差函数列表, 参数列表
            
        '''
        x0, y0, x1, y1, x2, y2 = sp.symbols("x0 y0 x1 y1 x2 y2")
        args = [x0, y0, x1, y1, x2, y2]
        line1 = maker_line_1(m, n)
        line2 = maker_line_2(x3, y3)
        line3 = maker_line_3(m, n, x3, y3)
        line4 = maker_line_4(m)
        parabola1 = maker_parabola_1(a, b)
        parabola2 = maker_parabola_2(a, b, c)
        parabola3 = maker_parabola_3(x3, y3)
        funcr = [line1, line2, line3, line4, parabola1, parabola2, parabola3]
        return funcr, args
    
    def plot_solve(final: list, x_0: tuple, m: float, n: float, a: float, b: float, c: float, x3: float, y3: float):
        '''
        Parameters
        ----------
        final : list
            最终迭代点列表
            
        x_0 : tuple
            初始点
            
        m : float
            直线的斜率
            
        n : float
            直线的截距
            
        a : float
            抛物线的二次项系数
            
        b : float
            抛物线的一次项系数
            
        c : float
            抛物线的常数项
            
        x3 : float
            所求圆须通过的定点的横坐标
            
        y3 : float
            所求圆须通过的定点的纵坐标
            

        Returns
        -------
        tuple :
            残差函数列表, 参数列表
            
        '''
        r = np.sqrt((x3 - final[0])**2 + (y3 - final[1])**2)
        x = np.linspace(final[0] - r, final[0] + r, 5000)
        y1 = np.sqrt(r**2 - (x - final[0])**2) + final[1]
        y2 = -np.sqrt(r**2 - (x - final[0])**2) + final[1]
        plt.plot(x, y1, c="teal", linestyle="dashed")
        plt.plot(x, y2, c="teal", linestyle="dashed")
        x1 = np.linspace(min(x) - 2, max(x) + 2, 5000)
        ln1, = plt.plot(x1, m*x1 + n, c="maroon", linestyle="dashed")
        ln2, = plt.plot(x1, a*x1**2 + b*x1 + c, c="orange", linestyle="dashed")
        x = sp.symbols("x")
        y1 = m*x + n
        y2 = a*x**2 + b*x + c
        plt.legend([ln1, ln2], [y1, y2])
        plt.annotate(r'$(sx_0, sy_0)$', xy=(x_0[0], x_0[1]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(sx_1, sy_1)$', xy=(x_0[2], x_0[3]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(sx_2, sy_2)$', xy=(x_0[4], x_0[5]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(x_0, y_0)$', xy=(final[0], final[1]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(x_1, y_1)$', xy=(final[2], final[3]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(x_2, y_2)$', xy=(final[4], final[5]), xycoords='data', xytext=(-20, +10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.annotate(r'$(x_3, y_3)$', xy=(x3, y3), xycoords='data', xytext=(+10, -10), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.axis("equal")
        plt.show()
        return None
    
    final = []
    funcr, args = maker_data(m, n, a, b, c, x3, y3)
    fin, _ = gauss_newton(funcr, args, x_0, False, epsilon=eps)
    for i in fin: # for i in map(round, fin):
        final.append(round(i, 2)) # final.append(i)
    print("(x0, y0)=",(final[0], final[1]),"\n(x1, y1)=",(final[2], final[3]),"\n(x2, y2)=",(final[4], final[5]))
    if draw is True: # 绘图
        plot_solve(final, x_0, m, n, a, b, c, x3, y3)
    return None

__all__ = [solution]