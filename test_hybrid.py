import optimtool.hybrid as oh
from optimtool.base import sp
x1, x2 = sp.symbols("x1 x2")
obf = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
# try to timely update next x_0 when reduce epsilon on objective to hybrid module, like 6e-2, 3e-2, 5e-5.
print(oh.nesterov.accer(obf, [x1, x2], (2, 3), verbose=True, proxim='ln', epsilon=4.00501))
# print(oh.fista.decline(obf, [x1, x2], (2, 3), verbose=True, proxim='ln', epsilon=4.00501))
# break criterion is chosen where the norm value of each gradient is less than epsilon in hybrid
# users need to set epsilon to a larger value when encounter `RecursionError: maximum recursion depth exceeded in comparison`
# x_0 was renewed with rate set by tk*epsilon in iteration proccess, tk*epsilon ranges from 1e-3 and `_proxim/ln` contribute the precision of `delta` within 1e-6 by tk*mu.