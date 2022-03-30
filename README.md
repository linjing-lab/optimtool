# Optimtool

[![Matplotlib Latest Release](https://img.shields.io/pypi/v/matplotlib.svg)](https://pypi.org/project/matplotlib/);&emsp;[![Numpy Latest Release](https://img.shields.io/pypi/v/numpy.svg)](https://pypi.org/project/numpy/);&emsp;[![Sympy Latest Release](https://img.shields.io/pypi/v/sympy.svg)](https://pypi.org/project/sympy/);&emsp;[![PyPI Latest Release](https://img.shields.io/pypi/v/optimtool.svg)](https://pypi.org/project/optimtool/);

Chinese blog homepage：https://blog.csdn.net/linjing_zyq

GitCode Url（It will not be updated, and the version will stay at 2.3.4）： [DeeGLMath / optimtool · GitCode](https://gitcode.net/linjing_zyq/optimtool)

How to use it：`pip install optimtool`

In this project, the solution types of problems corresponding to each method are different, so some problems that are not applicable to the method will appear in the actual application process. At this time, it can be solved by replacing the method library. All methods are designed and encapsulated on the basis of norm, derivative, convex set, convex function, conjugate function, subgradient and optimization theory. The algorithms of hybrid-optimization and the application fields of unconstrained-optimization and constrained-optimization are still being updated. It is hoped that more people in the industry will provide a good algorithm design framework.

## Introduce to use unconstrained-optimization

There are five mainstream methods, one of which is used to solve the nonlinear least squares problem.

### gradient_descent

`import packages`:

from optimtool.unconstrain import gradient_descent

`method list`:

| method                                                                                                                              | explanation                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| solve(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0)                                                              | The exact step size is solved by solving the equation.                                                  |
| steepest(funcs, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-10, k=0)                                           | The inexact step size is obtained by using the line search method.                                      |
| barzilar_borwein(funcs, args, x_0, draw=True, output_f=False, method="grippo", M=20, c1=0.6, beta=0.6, alpha=1, epsilon=1e-10, k=0) | The non monotonic line search method (grippo and Zhang hanger) is used to obtain the inexact step size. |

### newton

`import packages`:

from optimtool.unconstrain import newton

`method list`:

| method                                                                                          | explanation                                                                        |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| classic(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0)                        | The exact step size is obtained by directly solving the inverse of heather matrix. |
| modified(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-10, k=0) | The positive definiteness of heather matrix is determined and corrected.           |
| CG(funcs, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-6, k=0)              | The conjugate gradient method is used to solve the optimal step size.              |

### newton_quasi

`import packages`:

from optimtool.unconstrain import newton_quasi

`method list`:

| method                                                                                       | explanation                                                                        |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| bfgs(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-10, k=0)  | Update Heather matrix by BFGS.                                                     |
| dfp(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=20, epsilon=1e-4, k=0)    | Update Heather matrix by DFP.                                                      |
| L_BFGS(funcs, args, x_0, draw=True, output_f=False, method="wolfe", m=6, epsilon=1e-10, k=0) | Finite memory and double loop method are used to solve the updated Heather matrix. |

### nonlinear_least_square

`import packages`:

from optimtool.unconstrain import nonlinear_least_square

`method_list`:

| method                                                                                                                                               | explanation                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| gauss_newton(funcr, args, x_0, draw=True, output_f=False, method="wolfe", epsilon=1e-10, k=0)                                                        | Gauss Newton proposed a framework for solving nonlinear least squares problems, including QR decomposition and other operations. |
| levenberg_marquardt(funcr, args, x_0, draw=True, output_f=False, m=100, lamk=1, eta=0.2, p1=0.4, p2=0.9, gamma1=0.7, gamma2=1.3, epsilon=1e-10, k=0) | A framework for solving nonlinear least squares problems proposed by Levenberg Marquardt.                                        |

### trust_region

`import packages`:

from optimtool.unconstrain import trust_region

`method list`:

| method                                                                                                                                            | explanation                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| steihaug_CG(funcs, args, x_0, draw=True, output_f=False, m=100, r0=1, rmax=2, eta=0.2, p1=0.4, p2=0.6, gamma1=0.5, gamma2=1.5, epsilon=1e-6, k=0) | The truncated conjugate gradient method is used to search the gradient. |

## Introduce to use constrained-optimization

Here is the method library of inequality constraints, equality constraints and mixed equality constraints.

### equal

`import packages`:

from optimtool.constrain import equal

`method list`:

| method                                                                                                                                                | explanation                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| penalty_quadratic(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", sigma=10, p=2, epsilon=1e-4, k=0)                     | Design idea based on quadratic penalty function.                            |
| lagrange_augmented(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", lamk=6, sigma=10, p=2, etak=1e-4, epsilon=1e-6, k=0) | Design idea of solving frame based on Augmented Lagrange multiplier method. |

### unequal

`import packages`:

from optimtool.constrain import unequal

`method list`:

| method                                                                                                                                                                   | explanation                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| penalty_quadratic(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", sigma=10, p=0.4, epsilon=1e-10, k=0)                                     | Design idea based on quadratic penalty function.                            |
| penalty_interior_fraction(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", sigma=12, p=0.6, epsilon=1e-6, k=0)                              | Design idea of interior point method.                                       |
| lagrange_augmented(funcs, args, cons, x_0, draw=True, output_f=False, method="gradient_descent", muk=10, sigma=8, alpha=0.2, beta=0.7, p=2, eta=1e-1, epsilon=1e-4, k=0) | Design idea of solving frame based on Augmented Lagrange multiplier method. |

### mixequal

`import packages`:

from optimtool.constrain import mixequal

`method list`:

| method                                                                                                                                                                                               | explanation                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| penalty_quadratic(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", sigma=10, p=0.6, epsilon=1e-10, k=0)                                             | Design idea based on quadratic penalty function.                            |
| penalty_L1(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", sigma=1, p=0.6, epsilon=1e-10, k=0)                                                     | The design idea of L1 penalty function is adopted.                          |
| lagrange_augmented(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, method="gradient_descent", lamk=6, muk=10, sigma=8, alpha=0.5, beta=0.7, p=2, eta=1e-3, epsilon=1e-4, k=0) | Design idea of solving frame based on Augmented Lagrange multiplier method. |

## Introduce to use hybrid-optimization

The algorithm of this section will be updated in the future

## Introduce to use example

In this section, we will discuss the application of unconstrained, constrained and hybrid optimization in different fields.

### Lasso

You can search the specific form and solution method of lasso problem on the Internet. Here, the solution of the objective function of lasso problem is dealt with uniformly.

`import packages`:

from optimtool.example import example

`method list`:

| method                                                                                                  | explanation                                                                                                                           |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| gradient_descent(A, b, mu, args, x_0, draw=True, output_f=False, delta=10, alp=1e-3, epsilon=1e-2, k=0) | The idea of smoothing lasso function is used to optimize the objective function, and the gradient descent kernel is used to solve it. |
| subgradient(A, b, mu, args, x_0, draw=True, output_f=False, alphak=2e-2, epsilon=1e-3, k=0)             | The subgradient method is used to solve the problem that the absolute value function is not differentiable at the origin.             |

### WanYuan

`problem describe`：

Given slope and intercept of the linear equation, a quadratic equation given quadratic coefficient, a coefficient, a constant term, through a given point of the circle, the circle point over the required straight line tangent to the parabola and cut-off point and center of the circle.

> This section is different from other sections, because I am a student in the department of mathematics. The project issued by my tutor WanYuan is a problem closely related to curve research. At present, the curves under discussion are roughly circles, ellipses, parabolas, hyperbolas, implicit curves, etc. (specific project address:[linjing-lab/curve-research: The solvers for scientific research in curves. (github.com)](https://github.com/linjing-lab/curve-research)) this is a specific problem. At present, it is only a preliminary version of the algorithm. The kernel used is Gaussian Newton method to solve the minimum value problem of seven residual functions.

`import packages`:

from optimtool.example import WanYuan

`method list`:

| method                                                          | explanation                                             |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| gauss_newton(m, n, a, b, c, x3, y3, x_0, draw=False, eps=1e-10) | The algorithm is implemented by Gaussian Newton kernel. |

## Test

### 1. Unconstrained optimization algorithm performance comparison

```python
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x1**2 + x2**2 + x3**2 + x4**2 - 0.25)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (1, 2, 3, 4)

f_list = []
title = ["gradient_descent_barzilar_borwein", "newton_CG", "newton_quasi_L_BFGS", "trust_region_steihaug_CG"]
colorlist = ["maroon", "teal", "slateblue", "orange"]
_, _, f = oo.unconstrain.gradient_descent.barzilar_borwein(funcs, args, x_0, False, True)
f_list.append(f)
_, _, f = oo.unconstrain.newton.CG(funcs, args, x_0, False, True)
f_list.append(f)
_, _, f = oo.unconstrain.newton_quasi.L_BFGS(funcs, args, x_0, False, True)
f_list.append(f)
_, _, f = oo.unconstrain.trust_region.steihaug_CG(funcs, args, x_0, False, True)
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E6%97%A0%E7%BA%A6%E6%9D%9F%E4%BC%98%E5%8C%96%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

### 2. Nonlinear least squares problem

```python
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

r1, r2, x1, x2 = sp.symbols("r1 r2 x1 x2")
r1 = x1**3 - 2*x2**2 - 1
r2 = 2*x1 + x2 - 2
funcr = sp.Matrix([r1, r2])
args = sp.Matrix([x1, x2])
x_0 = (2, 2)

f_list = []
title = ["gauss_newton", "levenberg_marquardt"]
colorlist = ["maroon", "teal"]
_, _, f = oo.unconstrain.nonlinear_least_square.gauss_newton(funcr, args, x_0, False, True)
f_list.append(f)
_, _, f = oo.unconstrain.nonlinear_least_square.levenberg_marquardt(funcr, args, x_0, False, True)
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

### 3. Equality constrained optimization Test

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

f, x1, x2 = sp.symbols("f x1 x2")
f = x1 + np.sqrt(3) * x2
c1 = x1**2 + x2**2 - 1
funcs = sp.Matrix([f])
cons = sp.Matrix([c1])
args = sp.Matrix([x1, x2])
x_0 = (-1, -1)

f_list = []
title = ["penalty_quadratic", "lagrange_augmented"]
colorlist = ["maroon", "teal"]
_, _, f = oo.constrain.equal.penalty_quadratic(funcs, args, cons, x_0, False, True)
f_list.append(f)
_, _, f = oo.constrain.equal.lagrange_augmented(funcs, args, cons, x_0, False, True)
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

### 4. Inequality constrained optimization test

```python
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

f, x1, x2 = sp.symbols("f x1 x2")
f = x1**2 + (x2 - 2)**2
c1 = 1 - x1
c2 = 2 - x2
funcs = sp.Matrix([f])
cons = sp.Matrix([c1, c2])
args = sp.Matrix([x1, x2])
x_0 = (2, 3)

f_list = []
title = ["penalty_quadratic", "penalty_interior_fraction"]
colorlist = ["maroon", "teal"]
_, _, f = oo.constrain.unequal.penalty_quadratic(funcs, args, cons, x_0, False, True, method="newton", sigma=10, epsilon=1e-6)
f_list.append(f)
_, _, f = oo.constrain.unequal.penalty_interior_fraction(funcs, args, cons, x_0, False, True, method="newton")
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E4%B8%8D%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

`Single test Lagrange method`：

```python
import sympy as sp

# import optimtool
import optimtool as oo

# make functions
f1 = sp.symbols("f1")
x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
f1 = x1**2 + x2**2 + 2*x3**3 + x4**2 - 5*x1 - 5*x2 - 21*x3 + 7*x4
c1 = 8 - x1 + x2 - x3 + x4 - x1**2 - x2**2 - x3**2 - x4**2
c2 = 10 + x1 + x4 - x1**2 - 2*x2**2 - x3**2 - 2*x4**2
c3 = 5 - 2*x1 + x2 + x4 - 2*x1**2 - x2**2 - x3**2
cons_unequal1 = sp.Matrix([c1, c2, c3])
funcs1 = sp.Matrix([f1])
args1 = sp.Matrix([x1, x2, x3, x4])
x_1 = (0, 0, 0, 0)

x_0, _, f = oo.constrain.unequal.lagrange_augmented(funcs1, args1, cons_unequal1, x_1, output_f=True, method="trust_region", sigma=1, muk=1, p=1.2)
for i in range(len(x_0)):
     x_0[i] = round(x_0[i], 2)
print("\nfinal point：", x_0, "\nTarget function value：", f[-1])
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E6%9C%80%E9%80%9F%E4%B8%8B%E9%99%8D%E6%B3%95%E6%B5%8B%E8%AF%95%E5%9B%BE%E4%BE%8B.png"/></div>

`result`：

```python
final point： [ 2.5   2.5   1.87 -3.5 ] 
Target function value： -50.94151192711454
```

### 5. Mixed equation constraint test

```python
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

f, x1, x2 = sp.symbols("f x1 x2")
f = (x1 - 2)**2 + (x2 - 1)**2
c1 = x1 - 2*x2
c2 = 0.25*x1**2 - x2**2 - 1
funcs = sp.Matrix([f])
cons_equal = sp.Matrix([c1])
cons_unequal = sp.Matrix([c2])
args = sp.Matrix([x1, x2])
x_0 = (0.5, 1)

f_list = []
title = ["penalty_quadratic", "penalty_L1", "lagrange_augmented"]
colorlist = ["maroon", "teal", "orange"]
_, _, f = oo.constrain.mixequal.penalty_quadratic(funcs, args, cons_equal, cons_unequal, x_0, False, True)
f_list.append(f)
_, _, f = oo.constrain.mixequal.penalty_L1(funcs, args, cons_equal, cons_unequal, x_0, False, True)
f_list.append(f)
_, _, f = oo.constrain.mixequal.lagrange_augmented(funcs, args, cons_equal, cons_unequal, x_0, False, True)
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/%E6%B7%B7%E5%90%88%E7%AD%89%E5%BC%8F%E7%BA%A6%E6%9D%9F%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

### 6. Lasso problem test

```python
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
args = sp.Matrix(x)
x_0 = tuple([1 for i in range(8)])

f_list = []
title = ["gradient_descent", "subgradient"]
colorlist = ["maroon", "teal"]
_, _, f = oo.example.Lasso.gradient_descent(A, b, mu, args, x_0, False, True, epsilon=1e-4)
f_list.append(f)
_, _, f = oo.example.Lasso.subgradient(A, b, mu, args, x_0, False, True)
f_list.append(f)

# draw
handle = []
for j, z in zip(colorlist, f_list):
    ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
    handle.append(ln)
plt.xlabel("$Iteration \ times \ (k)$")
plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
plt.legend(handle, title)
plt.title("Performance Comparison")
plt.show()
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/Lasso%E8%A7%A3%E6%B3%95%E5%87%BD%E6%95%B0%E6%B5%8B%E8%AF%95.png"/></div>

### 7. WanYuan problem test

```python
# import packages
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

# make data
m = 1
n = 2
a = 0.2
b = -1.4
c = 2.2
x3 = 2*(1/2)
y3 = 0
x_0 = (0, -1, -2.5, -0.5, 2.5, -0.05)

# train
oo.example.WanYuan.gauss_newton(1, 2, 0.2, -1.4, 2.2, 2**(1/2), 0, (0, -1, -2.5, -0.5, 2.5, -0.05), draw=True)
```

<div align=center><img src="https://github.com/linjing-lab/optimtool/blob/master/visualization%20algorithms/WanYuan%E9%97%AE%E9%A2%98%E6%B5%8B%E8%AF%95%E5%9B%BE.png"/></div>
