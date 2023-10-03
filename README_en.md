<h1 align="center">optimtool</h1>

<p align="center">
<b>The fundamental package for scientific research in optimization.<sup><a href="https://github.com/linjing-lab/optimtool/tree/master/examples/doc">[?]</a></sup></b>
</p>

<p align='center'>
    <a href='https://www.oscs1024.com/cd/1527939640076619776?sign=522150ac'>
        <img src='https://www.oscs1024.com/platform/badge/linjing-lab/optimtool.svg?size=small' alt='OSCS Status' height='20'/>
    </a>
    <a href='https://pepy.tech/project/optimtool'>
        <img src="https://pepy.tech/badge/optimtool" alt="Total Downloads" height="20"/>
    </a>
    <a href='https://pepy.tech/project/optimtool'>
        <img src="https://pepy.tech/badge/optimtool/month" alt="Monthly Downloads" height="20"/> 
    </a>
    <a href='https://pepy.tech/project/optimtool'>
        <img src="https://pepy.tech/badge/optimtool/week" alt="Weekly Downloads" height="20"/> 
    </a>
</p>

[简体中文](./README.md) | English

## Introduction

&emsp;&emsp;optimtool adopts part of the theoretical framework in the book <Optimization: Modeling, Algorithms and Theory> published by Peking University Press, and [`Numpy`](https://github.com/numpy/numpy) is used to handle operations between numerical arrays efficiently. I skillfully applied .jacobian and other methods supported by [`Sympy`](https://github.com/sympy/sympy), and realized the switch from Sympy matrix to Numpy matrix combined with Python built-in functions dict and zip. A optimization toolbox for scientific research is designed for Researchers and Developers. Anyone can use a simple [`pip`](https://github.com/pypa/pip) command to download and use.

If you used **opimtool** in your research, welcome to cite it in your paper (follow the format below).

```text
林景. optimtool: The fundamental package for scientific research in optimization. 2021. https://pypi.org/project/optimtool/.
```

download latest version：
```text
git clone https://github.com/linjing-lab/optimtool.git
cd optimtool
pip install -e . --verbose
```
download stable version：
```text
pip install optimtool --upgrade
```
## Structure

```textile
|- optimtool
    |-- constrain
        |-- __init__.py
        |-- equal.py
        |-- mixequal.py
        |-- unequal.py
    |-- example
        |-- __init__.py
        |-- Lasso.py
        |-- WanYuan.py
    |-- hybrid
        |-- __init__.py
    |-- unconstrain
        |-- __init__.py
        |-- gradient_descent.py
        |-- newton.py
        |-- newton_quasi.py
        |-- nonlinear_least_square.py
        |-- trust_region.py  
    |-- __init__.py
    |-- _convert.py
    |-- _drive.py
    |-- _kernel.py
    |-- _search.py
    |-- _typing.py
    |-- _utils.py
    |-- _version.py
    |-- base.py
```
&emsp;&emsp;When solving the global or local convergence points of different objective functions, different methods for obtaining the convergence points will have different convergence efficiency and different scope of application. In addition, research methods in different fields are constantly proposed, modified, improved and expanded in the research process, so these methods have become what people now call "optimization methods". All internally supported algorithms in this project are designed and improved on the basis of basic methodologies such as norm, derivative, convex set, convex function, conjugate function, subgradient and optimization theory.

&emsp;&emsp;optimtool has built-in algorithms with good convergence efficiency and properties in unconstrained optimization fields, such as Barzilar-Borwein non-monotone gradient descent method, modified Newton-method, finite memory BFGS method, truncated conjugate gradient trust-region-method, Gauss Newton-method, as well as quadratic penalty function method, augmented Lagrangian method and other algorithms used to solve constrained optimization problems.

## Getting Started

### Unconstrained Optimization Algorithms(unconstrain)

```python
import optimtool.unconstrain as ou
ou.[Method Name].[Function Name]([Target Function], [Parameters], [Initial Point])
```

#### Gradient Descent Methods(gradient_descent)

```python
ou.gradient_descent.[Function Name]([Target Function], [Parameters], [Initial Point])
```

| head meathod                                                                                                                                 | explain                                   |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| solve(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-10, k: int=0) -> OutputType                                                             | Solve the exact step by solving the equation                      |
| steepest(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType                                           | Use line search method to solve imprecise step size (wolfe line search is used by default)         |
| barzilar_borwein(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="Grippo", c1: float=0.6, beta: float=0.6, M: int=20, eta: float=0.6, alpha: float=1, epsilon: float=1e-10, k: int=0) -> OutputType | Update the step size using the nonmonotonic line search method proposed by Grippo and Zhang Hanger |

#### Newton Methods(newton)

```python
ou.newton.[Function Name]([Target Function], [Parameters], [Initial Point])
```

| head meathod                                                                                             | explain                                |
| ----------------------------------------------------------------------------------------------- | --------------------------------- |
| classic(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-10, k: int=0) -> OutputType                        | The next step is obtained by directly inverting the second derivative matrix of Target Function (Heather matrix) |
| modified(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", m: int=20, epsilon: float=1e-10, k: int=0) -> OutputType | Revise the current Heather matrix to ensure its positive definiteness (only one correction method is connected at present)      |
| CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", eps: float=1e-3, epsilon: float=1e-6, k: int=0) -> OutputType              | Newton conjugate gradient method is used to solve the gradient (a kind of inexact Newton method)         |

#### Quasi Newton Methods(newton_quasi)

```pythonF
ou.newton_quasi.[Function Name]([Target Function], [Parameters], [Initial Point])
```

| head meathod                                                                                          | explain              |
| -------------------------------------------------------------------------------------------- | --------------- |
| bfgs(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType  | Updating Heiser Matrix by BFGS Method    |
| dfp(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType    | Updating Heiser Matrix by DFP Method     |
| L_BFGS(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", m: float=6, epsilon: float=1e-10, k: int=0) -> OutputType | Updating the Heiser Matrix of BFGS by Double Loop Method |

#### Nonlinear Least Square Methods(nonlinear_least_square)

```python
ou.nonlinear_least_square.[Function Name]([Target Function], [Parameters], [Initial Point])
```

| head meathod                                                                                                                                                  | explain                         |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| gauss_newton(funcr: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False,, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType                                                        | Gauss Newton's method framework, including OR decomposition and other operations     |
| levenberg_marquardt(funcr: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, lamk: float=1, eta: float=0.2, p1: float=0.4, p2: float=0.9, gamma1: float=0.7, gamma2: float=1.3, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType | Methodology framework proposed by Levenberg Marquardt |

#### Trust Region Methods(trust_region)

```python
ou.trust_region.[Function Name]([Target Function], [Parameters], [Initial Point])
```

| head meathod                                                                                                                                               | explain                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| steihaug_CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, r0: float=1, rmax: float=2, eta: float=0.2, p1: float=0.4, p2: float=0.6, gamma1: float=0.5, gamma2: float=1.5, epsk: float=1e-6, epsilon: float=1e-6, k: int=0) -> OutputType | Truncated conjugate gradient method is used to search step size in this method |

### Constrained Optimization Algorithms(constrain)

```python
import optimtool.constrain as oc
oc.[Method Name].[Function Name]([Target Function], [Parameters], [Equal Constraint Table], [Unequal Constraint Table], [Initial Point])
```

#### Equal Constraint(equal)

```python
oc.equal.[Function Name]([Target Function], [Parameters], [Equal Constraint Table], [Initial Point])
```

| head meathod                                                                                                                                                   | explain        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| penalty_quadratice(funcs: FuncArray, args: FuncArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10, p: float=2, epsk: float=1e-4, epsilon: float=1e-4, k: int=0) -> OutputType                     | Add secondary penalty    |
| lagrange_augmentede(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", lamk: float=6, sigma: float=10, p: float=2, etak: float=1e-4, epsilon: float=1e-6, k: int=0) -> OutputType | Augmented lagrange multiplier method |

#### Unequal Constraint(unequal)

```python
oc.unequal.[Function Name]([Target Function], [Parameters], [Unequal Constraint Table], [Initial Point])
```

| head meathod                                                                                                                                                                      | explain        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| penalty_quadraticu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10, p: float=0.4, epsk: float=1e-4, epsilon: float=1e-10, k: int=0) -> OutputType                                     | Add secondary penalty    |
| lagrange_augmentedu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", muk: float=10, sigma: float=8, alpha: float=0.2, beta: float=0.7, p: float=2, eta: float=1e-1, epsilon: float=1e-4, k: int=0) -> OutputType | Augmented lagrange multiplier method |

#### Mixequal Constraint(mixequal)

```python
oc.mixequal.[Function Name]([Target Function], [Parameters], [Equal Constraint Table], [Unequal Constraint Table], [Initial Point])
```

| head meathod                                                                                                                                                                                                  | explain        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| penalty_quadraticm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10, p: float=0.6, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType                                             | Add secondary penalty    |
| penalty_L1(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=1, p: float=0.6, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType                                                     | L1 exact penalty function method  |
| lagrange_augmentedm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", lamk: float=6, muk: float=10, sigma: float=8, alpha: float=0.5, beta: float=0.7, p: float=2, eta: float=1e-3, epsilon: float=1e-4, k: int=0) -> OutputType | Augmented lagrange multiplier method |

### Application of Methods(example)

```python
import optimtool.example as oe
```

#### Lasso Problem(Lasso)

```python
oe.Lasso.[Function Name]([Matrxi A], [Matrix b], [Factor mu], [Parameters], [Initial Point])
```

| head meathod                                                                                                     | explain               |
| ------------------------------------------------------------------------------------------------------- | ---------------- |
| gradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, delta: float=10, alp: float=1e-3, epsilon: float=1e-2, k: int=0) -> OutputType | Smoothing Lasso Function Method      |
| subgradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, alphak: float=2e-2, epsilon: float=1e-3, k: int=0) -> OutputType             | Sub gradient method Lasso: avoiding first order nondifferentiability |
| approximate_point(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType | Adjacent operator updating |

#### Curve Tangency Problem(WanYuan)

```python
oe.WanYuan.[Function Name]([The Slope of the Line], [Intercept of Line], [Coefficient of Quadratic Term], [Coefficient of Primary Term], [Constant Term], [Abscissa of Circle Center], [Vertical Coordinate of Circle Center], [Initial Point])
```

Problem Description：

```text
Given the slope and intercept of a straight line, given the coefficient of the quadratic term, the coefficient of the primary term and the constant term of a parabolic function. It is required to solve a circle with a given center, which is tangent to the parabola and straight line at the same time. If there is a feasible scheme, please provide the coordinates of the tangent point.
```

| head meathod                                                             | explain                   |
| --------------------------------------------------------------- | -------------------- |
| solution(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float, x_0: tuple, verbose: bool=False, draw: bool=False, eps: float=1e-10) -> str | Using Gauss Newton Method to solve the 7 Residual Functions Constructed |

### Hybrid Optimization Algorithms(hybrid)

```python
import optimtool.hybrid as oh
```

## LICENSE

[MIT LICENSE](./LICENSE)