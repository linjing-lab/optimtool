import optimtool.unconstrain as ou
from optimtool.base import sp, np
w11, w21, w31 = sp.symbols('w11 w21 w31', real=True)
w12, w22, w32 = sp.symbols('w12 w22 w32', real=True)
b1, b2, b3 = sp.symbols('b1 b2 b3', real=True)
X = np.array([
    [1.0, 1.0], 
    [1.5, 1.0],   
    [1.0, 1.5],   
    [4.0, 1.0],   
    [4.5, 1.0],   
    [4.0, 1.5],   
    [1.0, 4.0], 
    [1.5, 4.0],   
    [1.0, 4.5],   
])
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std
Y = np.zeros((X.shape[0], 3))
Y[:3, 0] = 1
Y[3:6, 1] = 1
Y[6:, 2] = 1
def symbolic_loss():
    loss = 0.0
    for i in range(len(Y)):
        x1, x2 = X[i]
        z1 = w11 * x1 + w12 * x2 + b1
        z2 = w21 * x1 + w22 * x2 + b2
        z3 = w31 * x1 + w32 * x2 + b3
        exp_sum = sp.exp(z1) + sp.exp(z2) + sp.exp(z3)
        p1 = sp.exp(z1) / exp_sum
        p2 = sp.exp(z2) / exp_sum
        p3 = sp.exp(z3) / exp_sum
        loss_expr = - (Y[i, 0] * sp.log(p1) + Y[i, 1] * sp.log(p2) + Y[i, 2] * sp.log(p3))
        loss += loss_expr
    return loss / len(Y)
f_sym = symbolic_loss()
ou.gradient_descent.barzilar_borwein(f_sym, [w11, w21, w31, w12, w22, w32, b1, b2, b3], [0, 0, 0, 0, 0, 0, 0, 0, 0], verbose=True, epsilon=1e-2)
