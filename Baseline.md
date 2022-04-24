# 最优化领域核心算法python实现

版权归属：@[武汉理工大学 林景](https://github.com/linjing-lab)

```python
# 导入numpy与sympy两个依赖包
%matplotlib widget # 便于保存图像（Jupyter lab）
import numpy as np
import sympy as sp

# 批量生成符号变量
s = sp.symbols('s0:10')

# 无约束优化算法应用到约束优化领域时需要随时更改停机准则
# 1. 前后两次梯度之差小于某个值
# 2. 迭代次数保证在某值以内
# 3. 函数值之差小于某个值
```

## Iteration visualization

### 1. visual interface for one algorithm

```python
import matplotlib.pyplot as plt
def function_f_x_k(funcs, args, x_0, mu=None):
    funcsv = np.array(funcs.subs(dict(zip(args, x_0)))).astype(np.float64)
    if mu is not None:
        for i in x_0:
            funcsv += mu * np.abs(i)
    return funcsv[0][0]

def function_plot_iteration(f, draw, method):
    if draw is True:
        plt.plot([i for i in range(len(f))], f, c='r', ls='--')
        plt.xlabel("$k$")
        plt.ylabel("$f(x_k)$")
        plt.title(method)
        plt.show()
    return None
```

### 2. performance comparison of 7 good unconstrained optimization algorithms

> 核心函数稍作修改，保证传出函数值列表，以及epsilon值需要稍作修改。算法绘图接口参数传入False。

```python
def plot_performance_algorithms(funcs, args, x_0, test_type):
    method_list = []
    color_list = []
    if test_type == "unconstrained":
        method_list=["gradient_descent_barzilar_borwein", "newton_CG", "newton_quasi_L_BFGS", "trust_region_steihaug_CG"]
        color_list=["maroon", "teal", "slateblue", "orange"]
    if test_type == "mixequal":
        method_list = ["penalty_quadratic_mixequal", "penalty_L1", "lagrange_augmented_mixequal"]
        color_list = ['maroon', "teal", "slateblue"]
    if test_type == "unequal":
        method_list = ["penalty_quadratic_unequal", "penalty_interior_fraction"]
        color_list = ['maroon', "teal"]
    if test_type == "equal":
        method_list = ["penalty_quadratic_equal", "lagrange_augmented_equal"]
        color_list = ['maroon', "teal"]
    handle = []
    for i, j in zip(method_list, color_list):
        _, _, f = eval(i)(funcs, args, x_0, False, True)
        print(i + ": success!")
        ln, = plt.plot([i for i in range(len(f))], f, c=j, marker='o', linestyle='dashed')
        handle.append(ln)
    plt.xlabel("$Iteration \ times \ (k)$")
    plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
    plt.legend(handle, method_list)
    plt.title("Performance Comparison")
    plt.show()
    return None
```

#### 2.1 Extended Penalty Function

$$
f(x)=\sum_{i=1}^{n-1}(x_i-1)^2+(\sum_{i=1}^{n}x_j^2-0.25)^2
$$

初始点：$x_0=(1,2, ...,n)^T$​，$f_{opt}=0$​

`test_data`：

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x1**2 + x2**2 + x3**2 + x4**2 - 0.25)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (1, 2, 3, 4)
```

#### 2.2 Extended Freudenstein & Roth Function

$$
f(x)=\sum_{i=1}^{n/2}[(-13+x_{2i-1}+((5-x_{2i})x_{2i}-2)x_{2i})^2+(-29+x_{2i-1}+((x_{2i}+1)x_{2i}-14)x_{2i})^2]
$$

初始点：$x_0=(0.5, -2, 0.5, -2, ...., 0.5, -2)^T$，$x^{*}=(5,4,5,4,...,5,4)^T$，$f_{opt}=0$

`test_data`：

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = (-13 + x1 + ((5 - x2)*x2 - 2)*x2)**2 + (-13 + x3 + ((5 - x4)*x4 - 2)*x4)**2 + (-29 + x1 + ((1 + x2)*x2 - 14)*x2)**2 + (-29 + x3 + ((1 + x4)*x4 - 14)*x4)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (0.5, -2, 0.5, -2)
```

#### 2.3 Extended Trigonometric Function

$$
f(x)=\sum_{i=1}^{n}[(n-\sum_{j=1}^{n}cos \ x_j)+i(1-cos \ x_i)-sin \ x_i]^2
$$

初始点：$x_0=(\frac{1}{n}, \frac{1}{n}, ..., \frac{1}{n})^T$，$f_{opt}=0$

`test_data`：

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = ((4 - (sp.cos(x1) + sp.cos(x2) + sp.cos(x3) + sp.cos(x4))) + 1*(1 - sp.cos(x1)) - sp.sin(x1))**2 + ((4 - (sp.cos(x1) + sp.cos(x2) + sp.cos(x3) + sp.cos(x4))) + 2*(1 - sp.cos(x2)) - sp.sin(x2))**2 + ((4 - (sp.cos(x1) + sp.cos(x2) + sp.cos(x3) + sp.cos(x4))) + 3*(1 - sp.cos(x3)) - sp.sin(x3))**2 + ((4 - (sp.cos(x1) + sp.cos(x2) + sp.cos(x3) + sp.cos(x4))) + 4*(1 - sp.cos(x4)) - sp.sin(x4))**2 
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (0.25, 0.25, 0.25, 0.25)
```

#### 2.4 Extended White & Holst Function

$$
f(x)=\sum_{i=1}^{n/2}[100(x_{2i}-x_{2i-1}^3)^2+(1-x_{2i-1})^2]
$$

初始点：$x_0=(-1.2,1,-1.2,1,...,-1.2,1)^T$，$x^{*}=(1,1,...,1)^T$，$f_{opt}=0$

`test_data`：

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = 100*(x2 - x1**3)**2 + (1 - x1)**2 + 100*(x4 - x3**3)**2 + (1 - x3)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (-1.2, 1,-1.2, 1)
```

### 3. performance comparison of 2 good constrained optimization algorithms

#### 3.1 Raydan 1 Function

$$
f(x)=\sum_{i=1}^{n} \frac{i}{100}(\exp(x_i)-x_i)
$$

初始点：$x_0=(1,1,...,1)^T$​，$x^{*}=(0,0,...,0)^T$​，$f_{opt}=\sum_{i=1}^{n}\frac{i}{100}$​。

`unconstrained_test_data`：

```python
f = sp.symbols("f")
x1, x2, x3, x4= sp.symbols("x1 x2, x3, x4")
f = (1 / 100)*(sp.exp(x1) - x1) + (1 / 50)*(sp.exp(x2) - x2) + (3 / 100)*(sp.exp(x3) - x3) + (1 / 25)*(sp.exp(x4) - x4)
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
x_0 = (1, 1, 1, 1)
# x_* = (0, 0, 0, 0)
```

`constrained_test_data`：

```python
f = sp.symbols("f")
x1, x2= sp.symbols("x1 x2")
f = (1 / 100)*(sp.exp(x1) - x1) + (1 / 50)*(sp.exp(x2) - x2)
c1 = x1
c2 = x2 - x1 - 1
c3 = - x2 - 0.5
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2])
cons_equal = sp.Matrix([c1])
cons_unequal = sp.Matrix([c2, c3])
x_0 = (1, 1)
# x_* = (0, 0)
```

## Unconstrained optimization

### 1. linear_search_algorithm

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点，搜索方向
> 
> （非单调方法还需要额外给定两个参数：当前迭代次数，迭代点列表）
> 
> 输出：最优步长

`example`：

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = 100*(x2-x1**2)**2 + (1-x1)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2])
x_0 = (-1, 1)
d = np.array([[1, 1]])
```

#### 1.1 armijo

```python
def linear_search_armijo(funcs, args, x_0, d, gamma=0.5, c=0.1):
    assert gamma > 0
    assert gamma < 1
    assert c > 0
    assert c < 1
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while True:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            break
        else:
            alpha = gamma * alpha
    return alpha
```

`调用格式`：

```python
linear_search_armijo(funcs, args, x_0, d)
```

`搜索结果`：

```python
0.00390625
```

#### 1.2 goldstein

```python
def linear_search_goldstein(funcs, args, x_0, d, c=0.1, alphas=0, alphae=10, t=1.2, eps=1e-3):
    assert c > 0
    assert c < 0.5
    assert alphas < alphae
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c*alpha*res0.dot(d.T):
            if f1 >= f0 + (1 - c)*alpha*res0.dot(d.T):
                break;
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
                if alphae < np.inf:
                    alpha = 0.5 * (alphas + alphae)
                else:
                    alpha = t * alpha
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas-alphae) < eps:
            break;
    return alpha
```

`调用格式`：

```python
linear_search_goldstein(funcs, args, x_0, d)
```

`搜索结果`：

```python
0.00390625
```

#### 1.3 wolfe

```python
def linear_search_wolfe(funcs, args, x_0, d, c1=0.3, c2=0.5, alphas=0, alphae=2, eps=1e-3):
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas < alphae
    alpha = 1
    res = funcs.jacobian(args)
    reps = dict(zip(args, x_0))
    f0 = np.array(funcs.subs(reps)).astype(np.float64)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while alphas < alphae:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        if f1 <= f0 + c1*alpha*res0.dot(d.T):
            res1 = np.array(res.subs(dict(zip(args, x)))).astype(np.float64)
            if res1.dot(d.T) >= c2*res0.dot(d.T):
                break;
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if np.abs(alphas-alphae) < eps:
            break;
    return alpha
```

`调用格式`：

```python
linear_search_wolfe(funcs, args, x_0, d)
```

`搜索结果`：

```python
0.00390625
```

#### 1.4 nonmonotonic（非单调）

> 非单调方法一般与程序配套使用，例如在梯度下降的`barzilar_borwein`的方法中。
> 
> ```python
> def gradient_descent_barzilar_borwein(funcs, args, x_0, draw=True, output_f=False, M=20, c1=0.6, beta=0.6, alpha=1, epsilon=1e-6, k=0):
>  assert M >= 0
>  assert alpha > 0
>  assert c1 > 0
>  assert c1 < 1
>  assert beta > 0
>  assert beta < 1
>  res = funcs.jacobian(args)
>  point = []
>  f = []
>  while True:
>      point.append(x_0)
>      reps = dict(zip(args, x_0))
>      f.append(function_f_x_k(funcs, x_0))
>      dk = - np.array(res.subs(reps)).astype(np.float64)
>      if np.linalg.norm(dk) >= epsilon:
>          # 此处替换为linear_search_nonmonotonic_Grippo 或者 linear_search_nonmonotonic_ZhangHanger
>          # alpha = linear_search_nonmonotonic_Grippo(funcs, args, x_0, dk, k, point, M, c1, beta, alpha)
>          alpha = linear_search_nonmonotonic_ZhangHanger(funcs, args, x_0, dk, k, point, c1, beta, alpha)
>          delta = alpha * dk[0]
>          x_0 = x_0 + delta
>          sk = delta
>          yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) + dk
>          if yk.all != 0:
>              alpha = sk.dot(sk.T) / sk.dot(yk.T)
>          k = k + 1
>      else:
>          break
>  function_plot_iteration(f, draw, "gradient_descent_barzilar_borwein_ZhangHanger")
>  if output_f is True:
>         return x_0, k, f
>     else:
>         return x_0, k
> ```

`调用格式`：

```python
gradient_descent_barzilar_borwein(funcs, args, (0, 0))
```

##### 1.4.1 Grippo

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点，搜索方向,，当前迭代次数，当前更新的迭代点列表，阈值M，常数c1，beta，初始步长）
> 
> 输出：最优步长

```python
def linear_search_nonmonotonic_Grippo(funcs, args, x_0, d, k, point, M, c1, beta, alpha):
    assert M >= 0
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while True:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        fk = - np.inf
        for j in range(min(k, M) + 1):
            fk = max(fk, np.array(funcs.subs(dict(zip(args, point[k-j])))).astype(np.float64))
        if f1 <= fk + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha
```

`迭代结果`：

```python
(array([4., 2.]), 14)
```

##### 1.4.2 ZhangHanger

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点，搜索方向,，当前迭代次数，当前更新的迭代点列表，常数c1，beta，初始步长）
> 
> 输出：最优步长

```python
def function_Q_k(eta, k):
    assert k >= 0
    if k == 0:
        return 1
    else:
        return eta * function_Q_k(eta, k-1) + 1

def function_C_k(funcs, args, point, eta, k):
    assert k >= 0
    if k == 0:
        return np.array(funcs.subs(dict(zip(args, point[0])))).astype(np.float64)
    else:
        return (1 / (function_Q_k(eta, k))) * (eta * function_Q_k(eta, k-1) * function_C_k(funcs, args, point, eta, k - 1) + np.array(funcs.subs(dict(zip(args, point[k])))).astype(np.float64))

def linear_search_nonmonotonic_ZhangHanger(funcs, args, x_0, d, k, point, c1, beta, alpha, eta=0.6):
    assert eta > 0
    assert eta < 1
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    reps = dict(zip(args, x_0))
    res = funcs.jacobian(args)
    res0 = np.array(res.subs(reps)).astype(np.float64)
    while True:
        x = x_0 + (alpha*d)[0]
        f1 = np.array(funcs.subs(dict(zip(args, x)))).astype(np.float64)
        Ck = function_C_k(funcs, args, point, eta, k)
        if f1 <= Ck + c1 * alpha * res0.dot(d.T):
            break
        else:
            alpha = beta * alpha
    return alpha
```

`迭代结果`：

```python
(array([4., 2.]), 13)
```

### 2. gradient_descent_algorithm

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点
> 
> 输出：（最优点，迭代次数）

`example`：

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = x1**2+2*x2**2-4*x1-2*x1*x2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2])
```

#### 2.1 solve

> 该方法不够稳定，主要通过解方程得到最优步长，很多情况步长无解或无最优解。

```python
def gradient_descent_solve(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0):
    res = funcs.jacobian(args)
    m = sp.symbols("m")
    arg = sp.Matrix([m])
    fx = []
    while True:
        reps = dict(zip(args, x_0))
        fx.append(function_f_x_k(funcs, args, x_0))
        dk = -np.array(res.subs(reps)).astype(np.float64)
        if np.linalg.norm(dk) >= epsilon:
            xt = x_0 + m * dk[0]
            f = funcs.subs(dict(zip(args, xt)))
            h = f.jacobian(arg)
            mt = sp.solve(h)
            x_0 = (x_0 + mt[m] * dk[0]).astype(np.float64)
            k = k + 1
        else:
            break
    function_plot_iteration(fx, draw, "gradient_descent_solve")
    if output_f is True:
        return x_0, k, fx
    else:
        return x_0, k
```

`调用格式`：

```python
gradient_descent_solve(funcs, args, (0, 0))
```

`迭代结果`：

```python
(array([4., 2.]), 72)
```

#### 2.2 barzilar_borwein

> 该方法会被用于约束优化问题中，由于其很好的稳定性。

```python
def gradient_descent_barzilar_borwein(funcs, args, x_0, draw=True, output_f=False, M=20, c1=0.6, beta=0.6, alpha=1, epsilon=1e-10, k=0):
    assert M >= 0
    assert alpha > 0
    assert c1 > 0
    assert c1 < 1
    assert beta > 0
    assert beta < 1
    res = funcs.jacobian(args)
    point = []
    f = []
    while True:
        point.append(x_0)
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        dk = - np.array(res.subs(reps)).astype(np.float64)
        if np.linalg.norm(dk) >= epsilon:
            fk = - np.inf
            for j in range(min(k, M)+1):
                fk = max(fk, np.array(funcs.subs(dict(zip(args, point[k-j])))).astype(np.float64))
            while np.array(funcs.subs(dict(zip(args, x_0 + alpha * dk[0])))).astype(np.float64) >= fk -  c1 * alpha * np.linalg.norm(dk)**2:
                alpha = beta * alpha
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = delta
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) + dk
            if sk.dot(yk.T) != 0:
                alpha = sk.dot(sk.T) / sk.dot(yk.T)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "gradient_descent_barzilar_borwein")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
gradient_descent_barzilar_borwein(funcs, args, (0, 0))
```

`迭代结果`：

```python
(array([4., 2.]), 12)
```

#### 2.3 Lasso

> 输入：参数表{A: ,b: , mu: }，sp.Matrix形式的参数表，初始点
> 
> 输出：（最优点，迭代次数）

##### 2.3.1 function

$$
\min \frac{1}{2} ||Ax-b||^2+\mu ||x||_1
$$

给定$A_{m \times n}$，$x_{n \times 1}$，$b_{m \times 1}$，正则化常数$\mu$。解决该无约束最优化问题，该问题目标函数一阶不可导。

##### 2.3.2 test_data

```python
import scipy.sparse as ss
f, A, b, mu = sp.symbols("f A b mu")
x = sp.symbols('x1:17')
m = 8
n = 16
u = (ss.rand(n, 1, 0.1)).toarray()
Mu = 1e-3
args = sp.Matrix(x)
parameters = {A: np.random.randn(m, n), b: np.random.randn(m, n).dot(u), mu: Mu}
x_0 = tuple([1 for i in range(16)])
```

##### 2.3.3 analysis

将$||x||_1$​​转化为一种可以求导的形式，例如转化成如下：

| 名称                         | 函数                                              |
| -------------------------- | ----------------------------------------------- |
| 光滑化LASOO问题                 | $\min f_{\delta}(x)=\frac{1}{2}$                |
| $L_{\delta}(x)$            | $\sum_{i=1}^{n}l_{\delta}(x_i)$                 |
| $l_{\delta}(x)$            | $\frac{1}{2 \delta} x^2 $                       |
| $\nabla f_{\delta}(x)$     | $A^T(Ax-b)+\mu \nabla L_{\delta}(x)$            |
| $(\nabla L_{\delta}(x))_i$ | $sign(x_i)$                                     |
| alpha                      | $alpha \leq \frac{1}{L}$                        |
| iteration                  | $x^{k+1} = x^k - alpha ·  \nabla f_{\delta}(x)$ |

##### 2.3.4 code

```python
def function_get_f_delta_gradient(resv, argsv, mu, delta):
    f = []
    for i, j in zip(resv, argsv):
        abs_args = np.abs(j)
        if abs_args > delta:
            if j > 0:
                f.append(i + mu * 1)
            elif j < 0:
                f.append(i - mu * 1)
        else:
            f.append(i + mu * (j / delta))
    return f[0]

def Lasso_gradient_decent(parameters, args, x_0, draw=True, output_f=False, delta=10, alp=1e-3, epsilon=1e-2, k=0):
    funcs = sp.Matrix([0.5*((parameters[A]*args - parameters[b]).T)*(parameters[A]*args - parameters[b])])
    res = funcs.jacobian(args)
    L = np.linalg.norm((parameters[A].T).dot(parameters[A])) + parameters[mu] / delta
    point = []
    f = []
    while True:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0, parameters[mu]))
        resv = np.array(res.subs(reps)).astype(np.float64)
        argsv = np.array(args.subs(reps)).astype(np.float64)
        g = function_get_f_delta_gradient(resv, argsv, parameters[mu], delta)
        alpha = alp
        assert alpha <= 1 / L
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0, parameters[mu]))
            break
    function_plot_iteration(f, draw, "Lasso_gradient_decent")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
Lasso_gradient_decent(parameters, args, x_0)
```

#### 2.4 `Tikhonov regularized model`

### 3. subgradient_algorithm

#### 3.1 Lasso

> 输入：参数表{A: , b: , mu: }，sp.Matrix形式的参数表，初始点
> 
> 输出：（最优点，迭代次数）

##### 3.1.1 analysis

| 名称          | 函数                                                                                     |
| ----------- | -------------------------------------------------------------------------------------- |
| subgradient | $g(x^k)=A^T(Ax^k-b)+\mu · a$​​​，$a=1, x^k>0 \ ;a \in [-1, 1], x^k=0 \ ;a = -1, x^k<0$​ |
| alpha       | $t^k = \frac{0.002}{\sqrt{k + 1}}$​​​​                                                 |
| iteration   | $x^{k+1}=x^k-t^k·g(x^k)$​                                                              |

##### 3.1.2 code

```python
def function_get_subgradient(resv, argsv, mu):
    f = []
    for i, j in zip(resv, argsv):
        if j > 0:
            f.append(i + mu * 1)
        elif j == 0:
            f.append(i + mu * (2 * np.random.random_sample() - 1))
        else:
            f.append(i - mu * 1)
    return f[0]


def Lasso_subgradient(parameters, args, x_0, draw=True, output_f=False, alphak=2e-2, epsilon=1e-3, k=0):
    funcs = sp.Matrix([0.5*((parameters[A]*args - parameters[b]).T)*(parameters[A]*args - parameters[b])])
    res = funcs.jacobian(args)
    point = []
    f = []
    while True:
        reps = dict(zip(args, x_0))
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0, parameters[mu]))
        resv = np.array(res.subs(reps)).astype(np.float64)
        argsv = np.array(args.subs(reps)).astype(np.float64)
        g = function_get_subgradient(resv, argsv, parameters[mu])
        alpha = alphak / np.sqrt(k + 1)
        x_0 = x_0 - alpha * g
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) <= epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
    function_plot_iteration(f, draw, "Lasso_subgradient")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
Lasso_subgradient(parameters, args, x_0)
```

#### 3.2 `Complement of positive definite matrix`

### 4. newton_algorithm

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点
> 
> 输出：（最优点，迭代次数）

`example`：

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = (1 - x1)**2 + 2*(x2 - x1**2)**2
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2])
```

#### 4.1 classic

> 该方法在海瑟矩阵不正定或者条件数过大时会不稳定。

```python
def newton_classic(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        hessian = np.array(hes.subs(reps)).astype(np.float64)
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - np.linalg.inv(hessian).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            x_0 = x_0 + dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_classic")        
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_classic(funcs, args, (0, 0))
```

`迭代结果`：

```python
(array([1., 1.]), 2)
```

#### 4.2 modified

```python
def function_modify_hessian(hessian, m, pk=1):
    l = hessian.shape[0]
    while True:
        values, _ = np.linalg.eig(hessian)
        flag = (all(values) > 0) & (np.linalg.cond(hessian) <= m)
        if flag:
            break
        else:
            hessian = hessian + pk * np.identity(l)
            pk = pk + 1
    return hessian

# 修正牛顿法
def newton_modified(funcs, args, x_0, draw=True, output_f=False, m=20, epsilon=1e-10, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        hess = np.array(hes.subs(reps)).astype(np.float64)
        hessian = function_modify_hessian(hess, m)
        dk = - np.linalg.inv(hessian).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = linear_search_wolfe(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_modified")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_modified(funcs, args, (0, 0))
```

`迭代结果`：

```python
(array([0.99999853, 0.99999696]), 22)
```

#### 4.3 imprecise（非精确牛顿法）

> 采用的是牛顿共轭梯度法（CG_gradient）实现的非精确牛顿法

```python
def function_CG_gradient(A, b, dk, epsilon=1e-6, k=0):
    rk = b.T - A.dot(dk)
    pk = rk
    while True:
        if np.linalg.norm(pk) < epsilon:
            break
        else:
            ak = (rk.T).dot(rk) / ((pk.T).dot(A)).dot(pk)
            dk = dk + ak * pk
            bk_down = (rk.T).dot(rk)
            rk = rk - ak * A.dot(pk)
            bk = (rk.T).dot(rk) / bk_down
            pk = rk + bk * pk
        k = k + 1
    return dk.reshape(1, -1), k

def newton_CG(funcs, args, x_0, draw=True, output_f=False, epsilon=1e-6, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    dk0 = np.zeros((args.shape[0], 1))
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        hess = np.array(hes.subs(reps)).astype(np.float64)
        # 采用共轭梯度法求解梯度
        dk, _ = function_CG_gradient(hess, - gradient, dk0)
        if np.linalg.norm(dk) >= epsilon:
            alpha = linear_search_wolfe(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_CG")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_CG(funcs, args, (0, 0))
```

`迭代结果`：

```python
(array([0.99999999, 0.99999998]), 6)
```

#### 4.4 `Logical regression`

##### 4.4.1 function

$$
\min_x l(x) = \frac{1}{m} \sum_{i=1}^{m}\ln(1+\exp(-b_ia_i^Tx))+\lambda||x||_2^2
$$

其中$b_i \in \{-1, 1\}$，$a_{n \times m}$，$m$代表样本个数，$n$​代表一个样本的特征数。这是一个二类分类的逻辑回归模型。

##### 4.4.2 analysis

| 名称              | 函数                                                                 |
| --------------- | ------------------------------------------------------------------ |
| $\lambda$       | $\frac{1}{m}$，$m$代表样本数                                             |
| $x$             | 一个样本，维数为$n \times 1$                                               |
| $\nabla l(x)$   | $- \frac{1}{m} \sum_{i=1}^{m}(1 - p_i(x))b_ia_i + 2 \lambda x$     |
| $\nabla^2 l(x)$ | $\frac{1}{m} \sum_{i=1}^{m}(1 - p_i(x))p_i(x)a_ia_i^T + 2 \lambda$ |
| $p_i(x)$        | $\frac{1}{1+\exp(-b_ia_i^Tx)}$                                     |
| $A$             | $[a_1, a_2, ..., a_m]^T \in R^{m \times n}$​                       |
| $b$             | $[b_1, b_2, ..., b_m]^T$                                           |
| $p(x)$          | $(p_1(x), p_2(x), ..., p_m(x))^T$                                  |
| $\nabla l(x)$   | $ - \frac{1}{m} A^T(b - b \odot p(x)) + 2 \lambda x$               |
| $\nabla^2 l(x)$ | $\frac{1}{m} A^TW(x)A + 2 \lambda$                                 |
| $W(x)$          | 由$\{p_i(x)(1-p_i(x))\}_{i=1}^{m}$生成的对角矩阵，$\odot$表示两个按分量的乘积。        |
| $dk$            | $\nabla^2 l(x) dk = - \nabla l(x)$                                 |

##### 4.4.3 test_data

### 5. newton_quasi_algorithm（拟牛顿法）

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点
> 
> 输出：（最优点，迭代次数）

`example`：

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 + 2*x3)**4 + 10*(x1 - x4)**4
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
```

#### 5.1 BFGS

```python
def newton_quasi_bfgs(funcs, args, x_0, draw=True, output_f=False, m=20, epsilon=1e-10, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(np.float64)
    hess = function_modify_hessian(hess, m)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - np.linalg.inv(hess).dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = linear_search_wolfe(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - np.array(res.subs(reps)).astype(np.float64)
            if yk.all != 0:
                hess = hess + (yk.T).dot(yk) / sk.dot(yk.T) - (hess.dot(sk.T)).dot((hess.dot(sk.T)).T) / sk.dot((hess.dot(sk.T)))
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_bfgs")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_quasi_bfgs(funcs, args, (1, 0, 1, 0))
```

`迭代结果`：

```python
(array([-2.89424685e-09,  2.89424685e-10, -8.82584212e-10, -8.82584212e-10]), 110)
```

#### 5.2 DFP

```python
def newton_quasi_dfp(funcs, args, x_0, draw=True, output_f=False, m=20, epsilon=1e-4, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    hess = np.array(hes.subs(dict(zip(args, x_0)))).astype(np.float64)
    hess = function_modify_hessian(hess, m)
    hessi = np.linalg.inv(hess)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        gradient = np.array(res.subs(reps)).astype(np.float64)
        dk = - hessi.dot(gradient.T)
        dk = dk.reshape(1, -1)
        if np.linalg.norm(dk) >= epsilon:
            alpha = linear_search_wolfe(funcs, args, x_0, dk)
            delta = alpha * dk[0]
            x_0 = x_0 + delta
            sk = alpha * dk
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - np.array(res.subs(reps)).astype(np.float64)
            if yk.all != 0:
                hessi = hessi - (hessi.dot(yk.T)).dot((hessi.dot(yk.T)).T) / yk.dot(hessi.dot(yk.T)) + (sk.T).dot(sk) / yk.dot(sk.T)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_dfp")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_quasi_dfp(funcs, args, (1, 0, 1, 0))
```

`迭代结果`：

```python
(array([-0.03098675,  0.00309623, -0.0033431 , -0.00324443]), 10)
```

#### 5.3 L_BFGS

> 目前是初步版本，时间复杂度仍需要优化

```python
def function_L_BFGS_double_loop(q, p, s, y, m, k, Hkm):
    istart1 = max(0, k - 1)
    iend1 = max(0, k - m - 1)
    istart2 = max(0, k - m)
    iend2 = max(0, k)
    alpha = np.empty((k, 1))
    for i in range(istart1, iend1, -1):
        alphai = p[i] * s[i].dot(q.T)
        alpha[i] = alphai
        q = q - alphai * y[i]
    r = Hkm.dot(q.T)
    for i in range(istart2, iend2):
        beta = p[i] * y[i].dot(r)
        r = r + (alpha[i] - beta) * s[i].T
    return - r.reshape(1, -1)

def newton_quasi_L_BFGS(funcs, args, x_0, draw=True, output_f=False, m=6, epsilon=1e-10, k=0):
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    l = hes.shape[0]
    f = []
    s = []
    y = []
    p = []
    gamma = []
    gamma.append(1)
    while True:
        reps = dict(zip(args, x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        Hkm = gamma[k] * np.identity(l)
        grad = np.array(res.subs(reps)).astype(np.float64)
        dk = function_L_BFGS_double_loop(grad, p, s, y, m, k, Hkm)
        if np.linalg.norm(dk) >= epsilon:
            alphak = linear_search_wolfe(funcs, args, x_0, dk)
            x_0 = x_0 + alphak * dk[0]
            if k > m:
                s[k - m] = np.empty((1, l))
                y[k - m] = np.empty((1, l))
            sk = alphak * dk
            s.append(sk)
            yk = np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64) - grad
            y.append(yk)
            pk = 1 / yk.dot(sk.T)
            p.append(pk)
            gammak = sk.dot(sk.T) / yk.dot(yk.T)
            gamma.append(gammak)
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "newton_quasi_L_BFGS")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
newton_quasi_L_BFGS(funcs, args, (1, 1, 1, 1))
```

`迭代结果`：

```python
(array([ 1.25128218e-08, -1.25128217e-09,  6.85128833e-09,  6.85128818e-09]), 247)
```

#### 5.4 `Kee Tracking`

#### 5.5 `Logical regression`

### 6. trust_region_algorithm（信赖域算法）

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， 初始点
> 
> 输出：（最优点，迭代次数）

`example`:

```python
f, x1, x2, x3, x4 = sp.symbols("f x1 x2 x3 x4")
f = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 + 2*x3)**4 + 10*(x1 - x4)**4
funcs = sp.Matrix([f])
args = sp.Matrix([x1, x2, x3, x4])
```

> 我采用的是截断共轭梯度法实现信赖域子问题的求解

#### 6.1 steihaug_CG

```python
def function_Eq_Sovle(sk, pk, delta):
    m = sp.symbols("m", positive=True)
    r = (sk + m * pk)[0]
    sub = 0
    for i in r:
        sub += i**2
    h = sp.sqrt(sub) - delta
    mt = sp.solve(h)
    return mt[0]

def function_steihaug_CG(sk, rk, pk, B, delta, epsilon=1e-3, k=0):
    s = []
    r = []
    p = []
    while True:
        s.append(sk)
        r.append(rk)
        p.append(pk)
        pbp = (p[k].dot(B)).dot(p[k].T)
        if pbp <= 0:
            m = function_Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        alphak = np.linalg.norm(r[k])**2 / pbp
        sk = s[k] + alphak * p[k]
        if np.linalg.norm(sk) > delta:
            m = function_Eq_Sovle(s[k], p[k], delta)
            ans = s[k] + m * p[k]
            break
        rk = r[k] + alphak * (B.dot(p[k].T)).T
        if np.linalg.norm(rk) < epsilon * np.linalg.norm(r[0]):
            ans = sk
            break
        betak = np.linalg.norm(rk)**2 / np.linalg.norm(r[k])**2
        pk = - rk + betak * p[k]
        k = k + 1
    return ans.astype(np.float64), k

# 信赖域算法
def trust_region_steihaug_CG(funcs, args, x_0, draw=True, output_f=False, m=100, r0=1, rmax=2, eta=0.2, p1=0.4, p2=0.6, gamma1=0.5, gamma2=1.5, epsilon=1e-6, k=0):
    assert eta >= 0
    assert r0 < rmax
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    res = funcs.jacobian(args)
    hes = res.jacobian(args)
    s0 = [0 for i in range(args.shape[0])]
    f = []
    while True:
        reps = dict(zip(args, x_0))
        funv = np.array(funcs.subs(reps)).astype(np.float64)
        f.append(funv[0][0])
        grad = np.array(res.subs(reps)).astype(np.float64)
        hessi = np.array(hes.subs(reps)).astype(np.float64)
        hessi = function_modify_hessian(hessi, m)
        dk, _ = function_steihaug_CG(s0, grad, - grad, hessi, r0)
        if np.linalg.norm(dk) >= epsilon:
            funvk = np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(np.float64)
            pk = (funv - funvk) / -(grad.dot(dk.T) + 0.5*((dk.dot(hessi)).dot(dk.T)))
            if pk < p1:
                r0 = gamma1 * r0
            else:
                if (pk > p2) | (np.linalg.norm(dk) == r0):
                    r0 = min(gamma2 * r0, rmax)
                else:
                    r0 = r0
            if pk > eta:
                x_0 = x_0 + dk[0]
            else:
                x_0 = x_0
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "trust_region_steihaug_CG")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
trust_region_steihaug_CG(funcs, args, (1, 0, 1, 0))
```

`迭代结果`：

```python
(array([ 0.00883332, -0.00088335,  0.00478148,  0.00478187]), 5591)
```

#### 6.2 `Logical regression`

### 7. nonlinear_least_square_algorithm（非线性最小二乘问题）

> 输入：sp.Matrix形式的残差函数， sp.Matrix形式的参数表， 初始点
> 
> 输出：（最优点，迭代次数）

`example`：

```python
r1, r2, x1, x2 = sp.symbols("r1 r2 x1 x2")
r1 = x1**3 - 2*x2**2 - 1
r2 = 2*x1 + x2 - 2
funcr = sp.Matrix([r1, r2])
args = sp.Matrix([x1, x2])
```

#### 7.1 gauss_newton

```python
def nonlinear_least_square_gauss_newton(funcr, args, x_0, draw=True, output_f=False, epsilon=1e-10, k=0):
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    f = []
    while True:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(np.float64)
        f.append(function_f_x_k(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(np.float64)
        q, r = np.linalg.qr(jk)
        dk = np.linalg.inv(r).dot(-(q.T).dot(rk)).reshape(1,-1)
        if np.linalg.norm(dk) > epsilon:
            alpha = linear_search_wolfe(funcs, args, x_0, dk)
            x_0 = x_0 + alpha * dk[0]
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "nonlinear_least_square_gauss_newton")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
nonlinear_least_square_gauss_newton(funcr, args, (2, 2))
```

`迭代结果`：

```python
(array([1.00000000e+00, 1.23094978e-13]), 5)
```

#### 7.2 levenberg_marquardt

```python
def nonlinear_least_square_levenberg_marquardt(funcr, args, x_0, draw=True, output_f=False, m=100, lamk=1, eta=0.2, p1=0.4, p2=0.9, gamma1=0.7, gamma2=1.3, epsilon=1e-10, k=0):
    assert eta >= 0
    assert eta < p1
    assert p1 < p2
    assert p2 < 1
    assert gamma1 < 1
    assert gamma2 > 1
    res = funcr.jacobian(args)
    funcs = sp.Matrix([(1/2)*funcr.T*funcr])
    resf = funcs.jacobian(args)
    hess = resf.jacobian(args)
    f = []
    while True:
        reps = dict(zip(args, x_0))
        rk = np.array(funcr.subs(reps)).astype(np.float64)
        f.append(function_f_x_k(funcs, args, x_0))
        jk = np.array(res.subs(reps)).astype(np.float64)
        dk = - (np.linalg.inv((jk.T).dot(jk) + lamk)).dot((jk.T).dot(rk))
        dk = dk.reshape(1, -1)
        pk_up = np.array(funcs.subs(reps)).astype(np.float64) - np.array(funcs.subs(dict(zip(args, x_0 + dk[0])))).astype(np.float64)
        grad_f = np.array(resf.subs(reps)).astype(np.float64)
        hess_f = np.array(hess.subs(reps)).astype(np.float64)
        hess_f = function_modify_hessian(hess_f, m)
        pk_down = - (grad_f.dot(dk.T) + 0.5*((dk.dot(hess_f)).dot(dk.T)))
        pk = pk_up / pk_down
        if np.linalg.norm(dk) >= epsilon:
            if pk < p1:
                lamk = gamma2 * lamk
            else:
                if pk > p2:
                    lamk = gamma1 * lamk
                else:
                    lamk = lamk
            if pk > eta:
                x_0 = x_0 + dk[0]
            else:
                x_0 = x_0
            k = k + 1
        else:
            break
    function_plot_iteration(f, draw, "nonlinear_least_square_levenberg_marquardt")        
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
nonlinear_least_square_levenberg_marquardt(funcs, args, (2, 2))
```

`迭代结果`：

```python
(array([1.00000000e+00, 1.97491835e-11]), 12)
```

#### 7.3 `Phase recovery`

## Constrained optimization

> 代码的稳定性仍需改进，无约束板块的内核可以改变。

`example for equal`：

$$
\min f(x), x \in R^n \\ s.t. c_i(x)=0,i=1,2,...,p
$$

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， sp.Matrix形式的等式约束函数表，初始点
> 
> 输出：（最优点，迭代次数）

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = x1 + np.sqrt(3) * x2
c1 = x1**2 + x2**2 - 1
funcs = sp.Matrix([f])
cons = sp.Matrix([c1])
args = sp.Matrix([x1, x2])
```

`example for unequal`：

$$
\min f(x),x \in R^n \\ s.t. c_i(x) \leq 0,i=1,2,...,p
$$

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， sp.Matrix形式的不等式约束函数表，初始点
> 
> 输出：（最优点，迭代次数）

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = x1**2 + (x2 - 2)**2
c1 = 1 - x1
c2 = 2 - x2
funcs = sp.Matrix([f])
cons = sp.Matrix([c1, c2])
args = sp.Matrix([x1, x2])
```

`example for mixequal`：

$$
\min f(x),x \in R^n \\ s.t. c_i(x)=0 ,i=1,2,...,p \ \\c_j(x) \leq 0, j=1,2,...,q
$$

> 输入：sp.Matrix形式的目标函数， sp.Matrix形式的参数表， sp.Matrix形式的等式约束函数表，sp.Matrix形式的不等式约束函数表， 初始点
> 
> 输出：（最优点，迭代次数）

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = (x1 - 2)**2 + (x2 - 1)**2
c1 = x1 - 2*x2
c2 = 0.25*x1**2 - x2**2 - 1
funcs = sp.Matrix([f])
cons_equal = sp.Matrix([c1])
cons_unequal = sp.Matrix([c2])
args = sp.Matrix([x1, x2])
```

### 1. penalty_algorithm

#### 1.1 quadratic

##### 1.1.1 equal

```python
def penalty_quadratic_equal(funcs, args, cons, x_0, draw=True, output_f=False, sigma=10, p=2, epsilon=1e-4, k=0):
    assert sigma > 0
    assert p > 1
    point = []
    sig = sp.symbols("sig")
    pen = funcs + (sig / 2) * cons.T * cons
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = pen.subs(sig, sigma)
        x_0, _ = gradient_descent_barzilar_borwein(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_equal")     
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
penalty_quadratic_equal(funcs, args, cons, (-1, -1))
```

`迭代结果`：

```python
(array([-0.50004882, -0.86610996]), 10)
```

##### 1.1.2 unequal

```python
def penalty_quadratic_unequal(funcs, args, cons, x_0, draw=True, output_f=False, sigma=1, p=0.4, epsilon=1e-10, k=0):
    assert sigma > 0
    assert p > 0
    assert p < 1
    point = []
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(np.float64)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons.T * consv])
        x_0, _ = gradient_descent_barzilar_borwein(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_unequal") 
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
penalty_quadratic_unequal(funcs, args, cons, (1, 1))
```

`迭代结果`：

```python
(array([9.99236167e-11, 2.00000000e+00]), 25)
```

##### 1.1.3 mixequal

```python
def penalty_quadratic_mixequal(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, sigma=1, p=0.6, epsilon=1e-10, k=0):
    assert sigma > 0
    assert p > 0
    point = []
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv = np.array(cons_unequal.subs(reps)).astype(np.float64)
        consv = np.where(consv <= 0, consv, 1)
        consv = np.where(consv > 0, consv, 0)
        pe = sp.Matrix([funcs + (sigma / 2) * cons_unequal.T * consv + (sigma / 2) * cons_equal.T * cons_equal])
        x_0, _ = gradient_descent_barzilar_borwein(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_quadratic_mixequal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
penalty_quadratic_mixequal(funcs, args, cons_equal, cons_unequal, (0.5, 1))
```

`迭代结果`：

```python
((2.0, 0.9999999999999999), 2)
```

#### 1.2 interior

> 需要保证**初始点在定义域内**，迭代过程中点有可能超出定义域，主要用于不等式约束，其中$\sigma$​​​称为惩罚因子。

##### 1.2.1 test_data

```python
f, x1, x2 = sp.symbols("f x1 x2")
f = x1**2 + 2*x1*x2 + x2**2 + 2*x1 - 2*x2
c1 = - x1
c2 = - x2
funcs = sp.Matrix([f])
cons = sp.Matrix([c1, c2])
args = sp.Matrix([x1, x2])
# x_* = (0, 1)
```

##### 1.2.2 fraction

$$
P_I(x,\sigma) = f(x) - \sigma \sum_{i \in I} \frac{1}{c_i(x)}
$$

```python
def penalty_interior_fraction(funcs, args, cons, x_0, draw=True, output_f=False, sigma=12, p=0.6, epsilon=1e-6, k=0):
    assert sigma > 0
    assert p > 0
    assert p < 1
    point = []
    f = []
    sub_pe = 0
    for i in cons:
        sub_pe += 1 / i
    sub_pe = sp.Matrix([sub_pe])
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = sp.Matrix([funcs - sigma * sub_pe])
        x_0, _ = newton_CG(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_interior_fraction")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
penalty_interior_fraction(funcs, args, cons, (0.5, 0.5))
```

`迭代结果`：

```python
((3.426526642696088e-06, 0.9999965735024556), 53)
```

##### 1.2.3 `log`

> 这里的代码不具有很好的通用性

$$
P_I(x,\sigma) = f(x) - \sigma \sum_{i \in I} \ln(-c_i(x))
$$

```python
def penalty_interior_log(funcs, args, cons, x_0, draw=True, output_f=False, sigma=2, p=0.4, epsilon=1e-3, k=0):
    assert sigma > 0
    assert p > 0
    assert p < 1
    point = []
    f = []
    sub_pe = 0
    for i in cons:
        sub_pe += sp.ln(- i)
    sub_pe = sp.Matrix([sub_pe])
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        pe = sp.Matrix([funcs - sigma * sub_pe])
        # 此处要修改
        x_0, _ = newton_CG(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_interior_log")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

#### 1.3 L1（精确罚函数）

> 主要用于一般约束

$$
P(x, \sigma)=f(x)+\sigma[\sum_{i \in \chi} |c_i(x)| + \sum_{i \in I} \overline{c_i(x)}]
$$

```python
def penalty_L1(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, sigma=1, p=0.6, epsilon=1e-10, k=0):
    assert sigma > 0
    assert p > 0
    point = []
    f = []
    while True:
        point.append(np.array(x_0))
        f.append(function_f_x_k(funcs, args, x_0))
        reps = dict(zip(args, x_0))
        consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
        consv_unequal = np.where(consv_unequal <= 0, consv_unequal, 1)
        consv_unequal = np.where(consv_unequal > 0, consv_unequal, 0)
        consv_equal = np.array(cons_equal.subs(reps)).astype(np.float64)
        consv_equal = np.where(consv_equal <= 0, consv_equal, 1)
        consv_equal = np.where(consv_equal > 0, consv_equal, -1)
        pe = sp.Matrix([funcs + sigma * cons_unequal.T * consv_unequal + sigma * cons_equal.T * consv_equal])
        x_0, _ = newton_CG(pe, args, tuple(x_0), draw=False)
        k = k + 1
        if np.linalg.norm(x_0 - point[k - 1]) < epsilon:
            point.append(np.array(x_0))
            f.append(function_f_x_k(funcs, x_0))
            break
        sigma = p * sigma
    function_plot_iteration(f, draw, "penalty_L1")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
penalty_L1(funcs, args, cons_equal, cons_unequal, (0.5, 1))
```

`迭代结果`：

```python
((1.9999998157717336, 1.0000003684565328), 31)
```

#### 1.4 `Lasso`

### 2. lagrange_augmented_algorithm（增广拉格朗日乘子法）

#### 2.1 equal

```python
def lagrange_augmented_equal(funcs, args, cons, x_0, draw=True, output_f=False, lamk=6, sigma=10, p=2, etak=1e-4, epsilon=1e-6, k=0):
    assert sigma > 0
    assert p > 1
    f = []
    lamk = np.array([lamk for i in range(cons.shape[0])]).reshape(cons.shape[0], 1)
    while True:
        L = sp.Matrix([funcs + (sigma / 2) * cons.T * cons + cons.T * lamk])
        f.append(function_f_x_k(funcs, args, x_0))
        x_0, _ = gradient_descent_barzilar_borwein(L, args, x_0, draw=False, epsilon=etak)
        k = k + 1
        reps = dict(zip(args, x_0))
        consv = np.array(cons.subs(reps)).astype(np.float64)
        if np.linalg.norm(consv) <= epsilon:
            f.append(function_f_x_k(funcs, x_0))
            break
        lamk = lamk + sigma * consv
        sigma = p * sigma
    function_plot_iteration(f, draw, "lagrange_augmented_equal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
lagrange_augmented_equal(funcs, args, cons, (0, 0))
```

`迭代结果`：

```python
(array([-0.5      , -0.8660254]), 5)
```

#### 2.2 mixequal

```python
def function_cons_unequal_L(cons_unequal, args, muk, sigma, x_0):
    sub = 0
    for i in range(cons_unequal.shape[0]):
        cons = muk[i] / sigma + cons_unequal[i]
        con = sp.Matrix([cons])
        conv = np.array(con.subs(dict(zip(args, x_0)))).astype(np.float64)
        if conv > 0:
            sub += cons**2 - muk[i]**2 / sigma**2
        else:
            sub += - muk[i]**2 / sigma**2
    sub = sp.Matrix([sub])
    return sub

def function_v_k(cons_equal, cons_unequal, args, muk, sigma, x_0):
    sub = 0
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_equal = np.array(cons_equal.subs(reps)).astype(np.float64)
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
    sub += (consv_equal.T).dot(consv_equal)
    for i in range(len_unequal):
        sub += (max(consv_unequal[i], - muk[i] / sigma))**2
    return np.sqrt(sub)

def function_renew_mu_k(cons_unequal, args, muk, sigma, x_0):
    reps = dict(zip(args, x_0))
    len_unequal = cons_unequal.shape[0]
    consv_unequal = np.array(cons_unequal.subs(reps)).astype(np.float64)
    for i in range(len_unequal):
        muk[i] = max(muk[i] + sigma * consv_unequal[i], 0)
    return muk

def lagrange_augmented_mixequal(funcs, args, cons_equal, cons_unequal, x_0, draw=True, output_f=False, lamk=6, muk=10, sigma=8, alpha=0.5, beta=0.7, p=2, eta=1e-3, epsilon=1e-4, k=0):
    assert sigma > 0
    assert p > 1
    assert alpha > 0 
    assert alpha <= beta
    assert beta < 1
    f = []
    lamk = np.array([lamk for i in range(cons_equal.shape[0])]).reshape(cons_equal.shape[0], 1)
    muk = np.array([muk for i in range(cons_unequal.shape[0])]).reshape(cons_unequal.shape[0], 1)
    while True:
        etak = 1 / sigma
        epsilonk = 1 / sigma**alpha
        cons_uneuqal_modifyed = function_cons_unequal_L(cons_unequal, args, muk, sigma, x_0)
        L = sp.Matrix([funcs + (sigma / 2) * (cons_equal.T * cons_equal + cons_uneuqal_modifyed) + cons_equal.T * lamk])
        f.append(function_f_x_k(funcs, args, x_0))
        x_0, _ = newton_CG(L, args, x_0, draw=False, epsilon=etak)
        k = k + 1
        vkx = function_v_k(cons_equal, cons_unequal, args, muk, sigma, x_0)
        if vkx <= epsilonk:
            res = L.jacobian(args)
            if (vkx <= epsilon) and (np.linalg.norm(np.array(res.subs(dict(zip(args, x_0)))).astype(np.float64)) <= eta):
                f.append(function_f_x_k(funcs, x_0))
                break
            else:
                lamk = lamk + sigma * np.array(cons_equal.subs(dict(zip(args, x_0)))).astype(np.float64)
                muk = function_renew_mu_k(cons_unequal, args, muk, sigma, x_0)
                sigma = sigma
                etak = etak / sigma
                epsilonk = epsilonk / sigma**beta
        else:
            lamk = lamk
            sigma = p * sigma
            etak = 1 / sigma
            epsilonk  = 1 / sigma**alpha
    function_plot_iteration(f, draw, "lagrange_augmented_mixequal")
    if output_f is True:
        return x_0, k, f
    else:
        return x_0, k
```

`调用格式`：

```python
lagrange_augmented_mixequal(funcs, args, cons_equal, cons_unequal, (0.5, 1))
```

`迭代结果`：

```python
(array([1.9999927, 1.0000146]), 7)
```

#### 2.3 convex

#### 2.4 kee_tracking

#### 2.5 semidefinite_programming

### 3. linear_programming_interior_algorithm

#### 3.1 primal_dual

#### 3.2 path_following

## Compound optimization

### 1. approximate_point_gradient_algorithm

### 2. Nesterov_accelerate_algorithm

#### 2.1 FISTA

#### 2.2 another_accelerate

### 3. approximate_point_algorithm

### 4. block_coordinate_descent_algorithm

### 5. dual_algorithm

### 6. alternating_direction_multiplier_algorithm

### 7. stochastic_optimization_algorithm
