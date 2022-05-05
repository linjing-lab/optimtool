# optimtool🔜

<div align="center">
    <img src="./asserts/logo.jpg">
</div>

<p align='center'>
    <a href='https://www.python.org/'>
        <img src="https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&color=2bbc8a">
    </a>
    <a href='https://numpy.org/'>
        <img src="https://img.shields.io/badge/Package-Numpy-informational?style=flat&logo=numpy&logoColor=white&color=2bbc8a">
    </a>
    <a href='https://www.sympy.org/en/index.html'>
        <img src="https://img.shields.io/badge/Package-Sympy-informational?style=flat&logo=sympy&logoColor=white&color=2bbc8a">
    </a>
</p>

如何下载： `pip install optimtool` 

[![PyPI Latest Release](https://img.shields.io/pypi/v/optimtool.svg)](https://pypi.org/project/optimtool/)

> 尽可能下载v2.3.4及以后（完备的导包方式与最新的无约束\约束方法库）
> 
> If you want to participate in the development, please follow the [baseline](./guides/baseline.md).
> 
> 如果你想参与开发，请遵循[baseline](./guides/baseline.md)。

## 项目介绍

&emsp;&emsp;optimtool采用了北京大学出版的《最优化：建模、算法与理论》这本书中的部分理论方法框架，运用了 [`Numpy`](https://github.com/numpy/numpy) 包高效处理数组间运算等的特性，巧妙地应用了 [`Sympy`](https://github.com/sympy/sympy) 内部支持的 .jacobian 等方法，并结合 Python 内置函数 dict 与 zip 实现了 Sympy 矩阵到 Numpy 矩阵的转换，最终设计了一个用于最优化科学研究领域的Python工具包。 研究人员可以通过简单的 [`pip`](https://github.com/pypa/pip) 指令进行下载与使用。 项目内无约束优化与约束优化板块的算法仍然需要不断更新、维护与扩充，并且应用于混合约束优化板块的算法将在日后上线。 我们非常欢迎广大热爱数学、编程的各界人士加入开发与更新最优化计算方法的队伍中，提出新的框架或算法，成为里程碑中的一员。

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
    |-- functions
        |-- __init__.py
        |-- linear_search.py
        |-- tools.py
    |-- hybrid
        |-- __init__.py
        |-- approximate_point_gradient.py
    |-- unconstrain
        |-- __init__.py
        |-- gradient_descent.py
        |-- newton.py
        |-- newton_quasi.py
        |-- nonlinear_least_square.py
        |-- trust_region.py  
    |-- __init__.py 
```
&emsp;&emsp;因为在求解不同的目标函数的全局或局部收敛点时，不同的求取收敛点的方法会有不同的收敛效率以及不同的适用范围，而且在研究过程中不同领域的研究方法被不断地提出、修改、完善、扩充，所以这些方法成了现在人们口中的`最优化方法`。 此项目中的所有内部支持的算法，都是在范数、导数、凸集、凸函数、共轭函数、次梯度和最优化理论等基础方法论的基础上进行设计与完善的。

&emsp;&emsp;optimtool内置了诸如Barzilar Borwein非单调梯度下降法、修正牛顿法、有限内存BFGS方法、截断共轭梯度法-信赖域方法、高斯-牛顿法等无约束优化领域收敛效率与性质较好的算法，以及用于解决约束优化问题的二次罚函数法、增广拉格朗日法等算法。

## [方法使用](./guides/methods.md)

## 方法测试

* [v2.3.4](./guides/tests(v2.3.4).md) ---> 完备的导包方式与无约束/约束优化方法库

* [v2.3.5](./guides/tests(v2.3.5).md) ---> 更加友好的输入参数形式

## [ISSUES](./guides/issues.md)

## [LICENSE](./LICENSE)