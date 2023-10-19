import optimtool.constrain as oc
from optimtool.base import sp
f, x1, x2 = sp.symbols("f x1 x2")
f = (x1 - 2)**2 + (x2 - 1)**2
c1 = x1 - x2 - 1
c2 = 0.25*x1**2 - x2 - 1
# oc.unequal.lagrange_augmentedu(f, (x1, x2), c2, (1.5, 0.5), verbose=True)
# oc.mixequal.lagrange_augmentedm(f, (x1, x2), c1, c2, (1.5, 0.5), verbose=True)
oc.equal.lagrange_augmentede(f, (x1, x2), c1, (1.5, 0.5), verbose=True, epsilon=1e-3)