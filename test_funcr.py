import optimtool.unconstrain as ou
from optimtool.base import sp, np
x = sp.symbols("x1:5")
res1 = 0.5*x[0] + 0.2*x[1] + 1.5*x[0]**3 - 2*x[2] # Damped oscillator with nonlinear term
res2 = 3*x[0] + x[1]**2 - x[2]**2 - 0.1*np.random.normal() # Coupled oscillator system
res3 = x[2]*(1 - x[3]) - x[0]*x[1] + 1.5*sp.sin(x[3]) # Predator-prey like interaction
res4 = x[3]*(x[0] - x[1]) + 0.5*sp.exp(-x[2]) - 2.0*x[1] # Delay differential equation component
ou.nonlinear_least_square.levenberg_marquardt([res1, res2, res3, res4], x, (1.0, 0.5, 0.2, 0.8), verbose=True, epsilon=1e-3)