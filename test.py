import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

f, x1, x2 = sp.symbols("f x1 x2")
f = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
c1 = -x1
c2 = -x2
funcs = f # [f] \ (f) \ sp.Matrix([f])
cons = [c1, c2] # (c1, c2) \ sp.Matrix([cons])
args = [x1, x2] # (x1, x2) \ sp.Matrix([args])
x_0 = (1, 2)

print(oo.constrain.unequal.penalty_interior_log(funcs, args, cons, x_0))