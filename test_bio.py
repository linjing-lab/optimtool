import optimtool.unconstrain as ou
from optimtool.base import sp, np

# Gene Regulatory Potential Function
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
vars = [x1, x2, x3]
f1 = (
    (x1**2) / (1 + x1**2)          # Hill type activation
    + sp.exp(-x2)                  # Expression attenuation
    + 0.5 * (x1 - x3)**2           # Gene coupling
    + 0.3 * sp.sin(5 * x2 * x3)    # nonlinear oscillation
)
x_0 = [0.8, 0.5, 1.2]
ou.trust_region.steihaug_CG(f1, vars, x_0, draw=True, verbose=True, epsk=1e-1, epsilon=1e-4)

# Michaelis-Menten
S = sp.symbols('S', real=True, positive=True)
Vmax, Km = sp.symbols('Vmax Km', real=True, positive=True)
v_pred = Vmax * S / (Km + S)
S_data = np.array([1, 2, 3, 4, 5])
v_data = np.array([0.9, 1.6, 2.1, 2.4, 2.7])
residuals = [v_pred.subs(S, Si) - vi for Si, vi in zip(S_data, v_data)]
f2 = sum(r**2 for r in residuals)
ou.newton_quasi.bfgs(f2, [Vmax, Km], [3.0, 1.0], verbose=True, epsilon=1e-4)

# Maximum Likelihood Estimation in Phylogenetics
t = sp.symbols('t', real=True, positive=True)
lam = sp.symbols('lam', real=True, positive=True)
P_same = (1/4) * (1 + 3*sp.exp(-4*lam*t/3))
P_diff = (1/4) * (1 - sp.exp(-4*lam*t/3))
n_same = 60
n_diff = 40
f3 = n_same * sp.log(P_same) + n_diff * sp.log(P_diff)
ou.gradient_descent.barzilar_borwein(f3, [t, lam], [1.0, 0.586], verbose=True, epsilon=1e-4)

# Protein Thermal Denaturation Model
delta_H = sp.symbols('Delta_H', real=True, positive=True)
Tm = sp.symbols('T_m', real=True, positive=True)
delta_Cp = sp.symbols('Delta_Cp', real=True)
T = 331.578947
delta_G = delta_H * (1 - T/Tm) - delta_Cp * ((Tm - T) + T * sp.log(T/Tm))
R = 8.314e-3
F_folded = 1 / (1 + sp.exp(delta_G / (R * T)))
sigma = sp.symbols('sigma', real=True, positive=True)
log_likelihood = -sp.log(1 / (sp.sqrt(2 * sp.pi) * sigma)) - (0.331567 - F_folded)**2 / (2 * sigma**2)
f4 = -log_likelihood
ou.newton.CG(f4, [delta_H, Tm, delta_Cp, sigma], [220.0, 330.0, 4.5, 0.02], verbose=True, epsilon=1e-4)