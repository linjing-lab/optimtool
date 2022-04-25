import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
funcs = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x1**2 + x2**2 + x3**2 + x4**2 - 0.25)**2
args = [x1, x2, x3, x4] # (x1, x2, x3, x4) \ sp.Matrix([args])
x_0 = (1, 2, 3, 4)

f_list = []
title = ["gradient_descent_barzilar_borwein", "newton_CG", "newton_quasi_L_BFGS", "trust_region_steihaug_CG"]
colorlist = ["maroon", "teal", "slateblue", "orange"]
_, _, fun = oo.unconstrain.gradient_descent.barzilar_borwein(funcs, args, x_0, False, True)
f_list.append(fun)
_, _, fun = oo.unconstrain.newton.CG(funcs, args, x_0, False, True)
f_list.append(fun)
_, _, fun = oo.unconstrain.newton_quasi.L_BFGS(funcs, args, x_0, False, True)
f_list.append(fun)
_, _, fun = oo.unconstrain.trust_region.steihaug_CG(funcs, args, x_0, False, True)
f_list.append(fun)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()