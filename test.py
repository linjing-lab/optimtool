import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

import scipy.sparse as ss
f, A, b, mu = sp.symbols("f A b mu")
x = sp.symbols('x1:9')
m = 4
n = 8
u = (ss.rand(n, 1, 0.1)).toarray()
A = np.random.randn(m, n)
b = A.dot(u)
mu = 1e-2
args = x # list(x) \ sp.Matrix(x)
x_0 = tuple([1 for i in range(8)])

print(oo.example.Lasso.subgradient(A, b, mu, args, x_0))