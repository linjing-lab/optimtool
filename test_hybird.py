import optimtool.hybrid as oh
from optimtool.base import sp
x1, x2 = sp.symbols("x1 x2")
obf = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
print(oh.nesterov.accer(obf, [x1, x2], (2, 3), lk=0.1, tk=0.2, verbose=True, proxim='ln', epsilon=10))
# break criterion is chosen where the norm value of each gradient is less than epsilon in hybird
# users need to set `epsilon` to a larger value in trials when encounter `RecursionError: `maximum recursion depth exceeded in comparison`

'''
$$
\min x^2+2xy+y^2+2x-2y \\
\mathrm{s.t.} x \geq 0, y \geq 0
$$
'''