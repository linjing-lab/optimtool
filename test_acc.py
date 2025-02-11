import optimtool
from optimtool.base import sp
from joblib import Parallel, delayed

x = sp.symbols("x1:5")

f1 = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2 + 100 * (x[3] - x[2]**2)**2 + (1 - x[2])**2
f2 = (sp.exp(x[0]) - x[0]) + (sp.exp(x[1]) - x[1]) + (sp.exp(x[2]) - x[2]) + (sp.exp(x[3]) - x[3])
f3 = 100 * (x[1] - x[0]**3)**2 + (1 - x[0])**2 + 100 * (x[3] - x[2]**3)**2 + (1 - x[2])**2
f4 = (x[0] - 1)**2 + (x[1] - 1)**2 + (x[2] - 1)**2 + ((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) - 0.25)**2
f5 = (sp.exp(x[0]) - x[0]) + (sp.exp(x[1]) - 2 * x[1]) + (sp.exp(x[2]) - 3 * x[2]) + (sp.exp(x[3]) - 4 * x[3])
# add fit function into acc_data with collections as functions, initial starts, epsilons for multiple convergence conditions. 
acc_data = [(f1, (-2, 2, -2, 2), 1e-1), (f2, (2, 2, 2, 2), 1e-2), (f3, (-1, 0.5, -1, 0.5), 1e-2), (f4, (5, 5, 5, 5), 1e-1), (f5, (0.5, 0.5, 0.5, 0.5), 1e-2)]

acc_function = optimtool.unconstrain.newton_quasi.bfgs

Parallel(n_jobs=-1, backend="loky", prefer="processes")(delayed(acc_function)(f[0], x, f[1], epsilon=f[2]) for f in acc_data)
# Parallel(n_jobs=-1, backend="threading", prefer="processes")(delayed(acc_function)(f, x, (10, 10, 10, 10), draw=False, epsilon=1e-4) for f in acc_data)