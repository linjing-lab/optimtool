# Gene Regulatory Potential Function
import optimtool.unconstrain as ou
from optimtool.base import sp
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
vars = [x1, x2, x3]
f = (
    (x1**2) / (1 + x1**2)          # Hill type activation
    + sp.exp(-x2)                  # Expression attenuation
    + 0.5 * (x1 - x3)**2           # Gene coupling
    + 0.3 * sp.sin(5 * x2 * x3)    # nonlinear oscillation
)
x_0 = [0.8, 0.5, 1.2]
ou.trust_region.steihaug_CG(f, vars, x_0, draw=True, verbose=True, epsk=1e-1, epsilon=1e-4)