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

简体中文 | [English](./README_en.md)

## 项目介绍

&emsp;&emsp;optimtool采用了北京大学出版的《最优化：建模、算法与理论》这本书中的部分理论方法框架，运用了 [`Numpy`](https://github.com/numpy/numpy) 包高效处理数组间运算等的特性，巧妙地应用了 [`Sympy`](https://github.com/sympy/sympy) 内部支持的 .jacobian 等方法，并结合 Python 内置函数 dict 与 zip 实现了 Sympy 矩阵到 Numpy 矩阵的转换，最终设计了一个用于最优化科学研究领域的Python工具包。 研究人员可以通过简单的 [`pip`](https://github.com/pypa/pip) 指令进行下载与使用。

如果你在研究中使用 **optimtool**，欢迎引用它在你的参考资料中（按照下面的格式）。

```text
林景. optimtool: The fundamental package for scientific research in optimization. 2021. https://pypi.org/project/optimtool/.
```

下载最新版：
```text
git clone https://github.com/linjing-lab/optimtool.git
cd optimtool
pip install -e . --verbose
```
下载稳定版：
```text
pip install optimtool --upgrade
```
下载没有更改库架构的版本：
```text
pip install optimtool==2.3.5
```
下载优化库架构并增加typing变量的版本：
```text
pip install optimtool>=2.4.0
```
下载改进h2h函数的版本：
```text
pip install optimtool>=2.4.2
```
下载增强文档表达和检测非法输入的版本：
```text
pip install optimtool>=2.5.0rc0
```
下载支持hybrid算法的版本：
```text
pip install optimtool>=2.5.0
```
下载更好地支持numpy的版本：
```text
pip install optimtool>=2.6.0
```
下载改进算法细节的版本：
```text
pip install optimtool>=2.7.0
```
> 建议：如果对自定义输入函数的需求比高版本中允许的输入类型检查更为强烈，推荐下载v2.4.4；如果需要在_convert.py和_typing.py中做可扩展的输入函数类型（基于sympy.core中所实现的类型），那么高版本中所作出的优化可以得到保留与体现。

## 项目结构

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
        |-- approt.py
        |-- fista.py
        |-- nesterov.py
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
    |-- _proxim.py
    |-- _search.py
    |-- _typing.py
    |-- _utils.py
    |-- _version.py
    |-- base.py
```
&emsp;&emsp;因为在求解不同的目标函数的全局或局部收敛点时，不同的求取收敛点的方法会有不同的收敛效率以及不同的适用范围，而且在研究过程中不同领域的研究方法被不断地提出、修改、完善、扩充，所以这些方法成了现在人们口中的`最优化方法`。 此项目中的所有内部支持的算法，都是在范数、导数、凸集、凸函数、共轭函数、次梯度和最优化理论等基础方法论的基础上进行设计与完善的。

&emsp;&emsp;optimtool内置了诸如Barzilar Borwein非单调梯度下降法、修正牛顿法、有限内存BFGS方法、截断共轭梯度法-信赖域方法、高斯-牛顿法等无约束优化领域收敛效率与性质好的算法，以及用于解决约束优化问题的二次罚函数法、增广拉格朗日法等算法。

## 开始使用

### 无约束优化算法（unconstrain）

```python
import optimtool.unconstrain as ou
ou.[方法名].[函数名]([目标函数], [参数表], [初始迭代点])
```

#### 梯度下降法（gradient_descent）

```python
ou.gradient_descent.[函数名]([目标函数], [参数表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                                                                          | 解释                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| solve(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-10, k: int=0) -> OutputType                                                                                                              | 通过解方程的方式来求解精确步长                        |
| steepest(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType                                                                                      | 使用线搜索方法求解非精确步长（默认使用wolfe线搜索）   |
| barzilar_borwein(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="Grippo", c1: float=0.6, beta: float=0.6, M: int=20, eta: float=0.6, alpha: float=1., epsilon: float=1e-10, k: int=0) -> OutputType | 使用Grippo与ZhangHanger提出的非单调线搜索方法更新步长 |

#### 牛顿法（newton）

```python
ou.newton.[函数名]([目标函数], [参数表], [初始迭代点])
```

| 方法头                                                                                                                                                                                               | 解释                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| classic(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-10, k: int=0) -> OutputType                                 | 通过直接对目标函数二阶导矩阵（海瑟矩阵）进行求逆来获取下一步的步长 |
| modified(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType           | 修正当前海瑟矩阵保证其正定性（目前只接入了一种修正方法）           |
| CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", eps: float=1e-3, epsilon: float=1e-6, k: int=0) -> OutputType | 采用牛顿-共轭梯度法求解梯度（非精确牛顿法的一种）                  |

#### 拟牛顿法（newton_quasi）

```python
ou.newton_quasi.[函数名]([目标函数], [参数表], [初始迭代点])
```

| 方法头                                                                                                                                                                                             | 解释                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| bfgs(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType             | BFGS方法更新海瑟矩阵       |
| dfp(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType              | DFP方法更新海瑟矩阵        |
| L_BFGS(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="wolfe", m: int=6, epsilon: float=1e-10, k: int=0) -> OutputType | 双循环方法更新BFGS海瑟矩阵 |

#### 非线性最小二乘法（nonlinear_least_square）

```python
ou.nonlinear_least_square.[函数名]([目标函数], [参数表], [初始迭代点])
```
| 方法头                                                                                                                                                                                                                                                                                                 | 解释                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| gauss_newton(funcr: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False,, draw: bool=True, output_f: bool=False, method: str="wolfe", epsilon: float=1e-10, k: int=0) -> OutputType                                                                                                        | 高斯-牛顿提出的方法框架，包括OR分解等操作 |
| levenberg_marquardt(funcr: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, lamk: float=1., eta: float=0.2, p1: float=0.4, p2: float=0.9, gamma1: float=0.7, gamma2: float=1.3, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType | Levenberg Marquardt提出的方法框架         |

#### 信赖域方法（trust_region）

```python
ou.trust_region.[函数名]([目标函数], [参数表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                                                                                                      | 解释                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| steihaug_CG(funcs: FuncArray, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, r0: float=1., rmax: float=2., eta: float=0.2, p1: float=0.4, p2: float=0.6, gamma1: float=0.5, gamma2: float=1.5, epsk: float=1e-6, epsilon: float=1e-6, k: int=0) -> OutputType | 截断共轭梯度法在此方法中被用于搜索步长 |

### 约束优化算法（constrain）

```python
import optimtool.constrain as oc
oc.[方法名].[函数名]([目标函数], [参数表], [等式约束表], [不等式约数表], [初始迭代点])
```

#### 等式约束（equal）

```python
oc.equal.[函数名]([目标函数], [参数表], [等式约束表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                                                                                  | 解释               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| penalty_quadratice(funcs: FuncArray, args: FuncArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10., p: float=2., epsk: float=1e-4, epsilon: float=1e-6, k: int=0) -> OutputType                 | 增加二次罚项       |
| lagrange_augmentede(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", lamk: float=6., sigma: float=10., p: float=2., etak: float=1e-4, epsilon: float=1e-6, k: int=0) -> OutputType | 增广拉格朗日乘子法 |

#### 不等式约束（unequal）

```python
oc.unequal.[函数名]([目标函数], [参数表], [不等式约束表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                                                                                                                   | 解释               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| penalty_quadraticu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10., p: float=0.4, epsk: float=1e-4, epsilon: float=1e-6, k: int=0) -> OutputType                                                  | 增加二次罚项       |
| lagrange_augmentedu(funcs: FuncArray, args: ArgArray, cons: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", muk: float=10., sigma: float=8., alpha: float=0.2, beta: float=0.7, p: float=2., eta: float=1e-1, epsilon: float=1e-4, k: int=0) -> OutputType | 增广拉格朗日乘子法 |

#### 混合等式约束（mixequal）

```python
oc.mixequal.[函数名]([目标函数], [参数表], [等式约束表], [不等式约束表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                                                                                                                                                                   | 解释               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| penalty_quadraticm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=10., p: float=0.6, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType                                                                  | 增加二次罚项       |
| penalty_L1(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", sigma: float=1., p: float=0.6, epsk: float=1e-6, epsilon: float=1e-10, k: int=0) -> OutputType                                                                           | L1精确罚函数法     |
| lagrange_augmentedm(funcs: FuncArray, args: ArgArray, cons_equal: FuncArray, cons_unequal: FuncArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, method: str="newton", lamk: float=6., muk: float=10., sigma: float=8., alpha: float=0.5, beta: float=0.7, p: float=2., etak: float=1e-3, epsilon: float=1e-4, k: int=0) -> OutputType | 增广拉格朗日乘子法 |

### 方法的应用（example）

```python
import optimtool.example as oe
```

#### Lasso问题（Lasso）

```python
oe.Lasso.[函数名]([矩阵A], [矩阵b], [因子mu], [参数表], [初始迭代点])
```

| 方法头                                                                                                                                                                                                                   | 解释                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| gradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, delta: float=10., alp: float=1e-3, epsilon: float=1e-3, k: int=0) -> OutputType | 光滑化Lasso函数法           |
| subgradient(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, alphak: float=2e-2, epsilon: float=1e-3, k: int=0) -> OutputType             | 次梯度法Lasso避免一阶不可导 |
| approximate_point(A: NDArray, b: NDArray, mu: float, args: ArgArray, x_0: PointArray, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-3, k: int=0) -> OutputType                           | 邻近算子更新                |

#### 曲线相切问题（WanYuan）

```python
oe.WanYuan.[函数名]([直线的斜率], [直线的截距], [二次项系数], [一次项系数], [常数项], [圆心横坐标], [圆心纵坐标], [初始迭代点])
```

问题描述：

```text
给定直线的斜率和截距，给定一个抛物线函数的二次项系数，一次项系数与常数项。 要求解一个给定圆心的圆，该圆同时与抛物线、直线相切，若存在可行方案，请给出切点的坐标。
```

| 方法头                                                                                                                                                       | 解释                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| solution(m: float, n: float, a: float, b: float, c: float, x3: float, y3: float, x_0: tuple, verbose: bool=False, draw: bool=False, eps: float=1e-10) -> str | 使用高斯-牛顿方法求解构造的7个残差函数 |

### 混合优化算法（hybrid）

```python
import optimtool.hybrid as oh
```

#### 近似点算法（approt）

```python
oh.approt.[函数名]([目标函数], [参数表], [初始迭代点], [正则化参数], [邻近算子名])
```

| 方法头                                                                                                                                                                                                             | 解释                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| grad(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType | 基于梯度方法的邻近近似 |

#### FISTA算法（fista）

```python
oh.fista.[函数名]([目标函数], [参数表], [初始迭代点], [正则化参数], [邻近算子名])
```

| 方法头                                                                                                                                                                                                                | 解释                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| normal(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType  | 两步计算一个新点       |
| variant(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType | normal法的等价变形     |
| decline(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType | 基于函数下降趋势的变体 |

#### Nesterov算法（nesterov）

```python
oh.nesterov.[函数名]([目标函数], [参数表], [初始迭代点], [正则化参数], [邻近算子名])
```

| 方法头                                                                                                                                                                                                                              | 解释                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| seckin(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType                | 第二类Nesterov加速法   |
| accer(funcs: FuncArray, args: ArgArray, x_0: PointArray, mu: float=1e-3, proxim: str="L1", lk: float=0.01, tk: float=0.02, verbose: bool=False, draw: bool=True, output_f: bool=False, epsilon: float=1e-4, k: int=0) -> OutputType | 复合优化算法的加速框架 |

## LICENSE

[MIT LICENSE](./LICENSE)