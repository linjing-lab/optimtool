import optimtool.constrain as oc
from optimtool.base import sp, np
x, y, z = sp.symbols('x y z', real=True)
# Objective function: Non convex function with rotation term
f = (x**2 + y**2 - 1)**2 + (x - y)**2 + sp.sin(3*x) + sp.cos(2*y) + 0.1*z**2
# Constraint 1: Inside the ellipsoid
g1 = x**2/4 + y**2/9 + z**2/16 - 1
# Constraint 2: Nonlinear wavy constraint
g2 = sp.sin(x) + sp.cos(y) + 0.5*z - 0.2
# Constraint 3: Hyperbolic Parabolic Constraint
g3 = x**2 - y**2 - z - 0.5
# Constraint 4: Circular Constraint
g4 = (x**2 + y**2 - 2)**2 + z**2 - 1.5
# Constraint 5: Exponential Constraint
g5 = sp.exp(-(x-1)**2 - (y+1)**2) + 0.1*z - 0.8
# All constraints (standard form: g_i(x) <= 0）
constraints = [g1, g2, g3, g4, g5]
final, _ = oc.unequal.penalty_quadraticu(funcs=f, args=[x, y, z], cons=constraints, x_0=[0.5, 0.5, 0.5], verbose=True, epsilon=1e-6)
for i in constraints:
    reps = dict(zip([x,y,z], final))
    consv = np.array(i.subs(reps)).astype(np.float64)
    assert consv <= 0
    print(i, consv) # check g_i(x) <= 0
        