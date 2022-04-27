import sympy as sp
import optimtool as oo

x1, x2 = sp.symbols("x1 x2")
f = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
c1 = - x1
c2 = - x2
args = [x1, x2] # (x1, x2) \ sp.Matrix([args])
x_0 = (2, 3)

print(oo.constrain.unequal.penalty_interior_log(f, args, [c1, c2], x_0))