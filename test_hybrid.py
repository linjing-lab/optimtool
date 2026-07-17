import optimtool.hybrid as oh
from optimtool.base import np, sp
x1, x2 = sp.symbols("x1 x2")
obf = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2 # should be constraint optimization with semi-positive values, solved by hybrid proxim='ln' method.
# try to timely update next x_0 when reduce epsilon on objective to hybrid module, like 6e-2, 3e-2, 5e-5.
print(oh.nesterov.accer(obf, [x1, x2], (2, 3), verbose=True, proxim='ln', epsilon=4.00501))
# print(oh.fista.decline(obf, [x1, x2], (2, 3), verbose=True, proxim='ln', epsilon=4.00501))
# break criterion is chosen where the norm value of each gradient is less than epsilon in hybrid
# users need to set epsilon to a larger value when encounter `RecursionError: maximum recursion depth exceeded in comparison`
# x_0 was renewed with rate set by tk*epsilon in iteration proccess, tk*epsilon ranges from 1e-3 and `_proxim/ln` contribute the precision of `delta` within 1e-6 by tk*mu.

x = sp.symbols('x')
w = sp.symbols('w1:6')
phi = [
    sp.exp(-((x - 0.2)**2) / 0.01),
    sp.exp(-((x - 0.4)**2) / 0.01),
    sp.exp(-((x - 0.6)**2) / 0.01),
    sp.exp(-((x - 0.8)**2) / 0.01),
    sp.sin(sp.pi * x)
]
x_obs = [0.25, 0.5, 0.75]
u_obs = [0.12, 0.18, 0.15]
data_fitting = sum(
    (sum(w[i] * phi[i].subs(x, xi) for i in range(5)) - ui)**2
    for xi, ui in zip(x_obs, u_obs)
)
print(oh.fista.variant(data_fitting, list(w), [1.0, 1.0, 1.0, 1.0, 1.0], verbose=True, proxim='L1', epsilon=1e-1))
# print(oh.fista.variant(data_fitting, list(w), [1.0, 1.0, 1.0, 1.0, 1.0], verbose=True, proxim='L2', epsilon=1e-1))