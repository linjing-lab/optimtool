import optimtool.unconstrain as ou
from optimtool.base import sp
x = sp.symbols("x1:5")
f = 100 * (x[1] - x[0]**2)**2 + \
    (1 - x[0])**2 + \
    100 * (x[3] - x[2]**2)**2 + \
    (1 - x[2])**2
x_0 = (-2, 0.5, -2, 0.5)
lbfgs = ou.newton_quasi.L_BFGS
lbfgs(f, x, x_0, verbose=True, draw=False, m=2)# m is history length as [k-m, k-1] in k iterations.
# lbfgs(f, x, x_0, verbose=True, draw=False, m=6)