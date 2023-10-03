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

from ..base import np, sp, plt

from .._typing import FuncType, Tuple, FuncArray, ArgArray

__all__ = ["solution"]

def solution(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float, x_0: tuple, verbose: bool=False, draw: bool=False, eps: float=1e-10) -> str:
    '''
    :param m: float, the slope of the linear equation.
    :param n: float, the intercept of linear equation.
    :param a: float, the quadratic coefficient of a parabola.
    :param b: float, the coefficient of the first order term of a parabola.
    :param c: float, the constant coefficient of a parabola.
    :param x3: float, the abscissa of the fixed point through which the circle must pass.
    :param y3: float, the ordinate of the fixed point through which the circle must pass.
    :param x_0: tuple, the initial point of the tangent point coordinate obtained.
    :param draw: bool, whether to plot a visual image. default: bool=False.
    :param eps: float, precision to stop train with inner kernel. default: float=1e-10.

    :return: returns a str result with final converged points.
    '''
    from ..unconstrain.nonlinear_least_square import gauss_newton
    def maker_line_1(m: float, n: float) -> FuncType:
        x1, y1 = sp.symbols("x1 y1")
        return m*x1 + n - y1

    def maker_line_2(x3: float, y3: float) -> FuncType:
        x1, y1, x0, y0 = sp.symbols("x1 y1 x0 y0")
        return (x1 - x0)**2 + (y1 - y0)**2 - ((x3 - x0)**2 + (y3 - y0)**2)

    def maker_line_3(m: float, n: float, x3: float, y3: float) -> FuncType:
        x0, y0= sp.symbols("x0 y0")
        return (2*m*n - 2*x0 - 2*m*y0)**2 - 4*(m**2 + 1)*(x0**2 + y0**2 - (x3 - x0)**2 + (y3 - y0)**2 + n**2 - 2*n*y0)

    def maker_line_4(m: float) -> FuncType:
        x1, y1, x0, y0= sp.symbols("x1 y1 x0 y0")
        return m*y1 - m*y0 + x1 - x0

    def maker_parabola_1(a: float, b: float) -> FuncType:
        x2, y2, x0, y0 = sp.symbols("x2 y2 x0 y0")
        return 2*a*x2*y2 - 2*a*x2*y0 + b*y2 - b*y0 - x0 + x2

    def maker_parabola_2(a: float, b: float, c: float) -> FuncType:
        x2, y2 = sp.symbols("x2 y2")
        return a*x2**2 + b*x2 + c - y2

    def maker_parabola_3(x3: float, y3: float) -> FuncType:
        x2, y2, x0, y0 = sp.symbols("x2 y2 x0 y0")
        return (x2 - x0)**2 + (y2 - y0)**2 - ((x3 - x0)**2 + (y3 - y0)**2)

    def maker_data(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float) -> Tuple[FuncArray, ArgArray]:
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
    fin, _ = gauss_newton(funcr, args, x_0, verbose=verbose, draw=False, epsilon=eps)
    for i in fin: # for i in map(round, fin):
        final.append(round(i, 2)) # final.append(i)
    if draw is True:
        plot_solve(final, x_0, m, n, a, b, c, x3, y3)
    return "(x0, y0)={}, (x1, y1)={}, (x2, y2)={}".format((final[0], final[1]), (final[2], final[3]), (final[4], final[5]))