import optimtool.unconstrain as ou
from optimtool.base import sp, np
w1, w2, w3, b = sp.symbols('w1 w2 w3 b', real=True)
np.random.seed(0)
n = 20 # samples
x1 = np.linspace(-3, 3, n)
x2 = 0.8 * x1 + np.random.randn(n) * 0.5
X = np.column_stack([x1, x2])
y_true = 2 * x1 - 3 * x2 + 1.5 * x1**2
noise = np.random.randn(n)
y = y_true + noise
def symbolic_loss():
    loss = 0
    for i in range(len(y)):
        x1_, x2_ = X[i]
        y_pred = w1 * x1_ + w2 * x2_ + w3 * x1_**2 + b
        loss += (y_pred - y[i]) ** 2
    return loss / len(y)
f_sym = symbolic_loss()
ou.newton_quasi.bfgs(f_sym, [w1, w2, w3, b], [0, 0, 0, 0], verbose=True)