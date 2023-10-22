import optimtool.unconstrain as ou
from optimtool.base import sp
x = sp.symbols("x1:5")
f = 100 * (x[1] - x[0]**2)**2 + \
    (1 - x[0])**2 + \
    100 * (x[3] - x[2]**2)**2 + \
    (1 - x[2])**2
x_0 = (-1.2, 1, -1.2, 1)
lbfgs = ou.newton_quasi.L_BFGS
lbfgs(f, x, x_0, verbose=True, draw=False, m=2)
# lbfgs(f, x, x_0, verbose=True, draw=False, m=6)