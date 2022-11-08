# 无约束优化工具箱示例


```python
# import packages
%matplotlib inline
import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo
```


```python
def train(funcs, args, x_0):
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
    return colorlist, f_list, title
```


```python
# 可视化函数：传参接口（颜色列表，函数值列表，标题列表）
def test(colorlist, f_list, title):
    handle = []
    for j, z in zip(colorlist, f_list):
        ln, = plt.plot([i for i in range(len(z))], z, c=j, marker='o', linestyle='dashed')
        handle.append(ln)
    plt.xlabel("$Iteration \ times \ (k)$")
    plt.ylabel("$Objective \ function \ value: \ f(x_k)$")
    plt.legend(handle, title)
    plt.title("Performance Comparison")
    return None
```

## Extended Freudenstein & Roth function

$$
f(x)=\sum_{i=1}^{n/2}(-13+x_{2i-1}+((5-x_{2i})x_{2i}-2)x_{2i})^2+(-29+x_{2i-1}+((x_{2i}+1)x_{2i}-14)x_{2i})^2, x_0=[0.5, -2, 0.5, -2, ..., 0.5, -2].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + \
    (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + \
    (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + \
    (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2
x_0 = (1, -1, 1, -1) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/8387aab0a8e4401db77021d6489d4481.png#pic_center" >

    


## Extended Trigonometric function:

$$
f(x)=\sum_{i=1}^{n}((n-\sum_{j=1}^{n}\cos x_j)+i(1-\cos x_i)-\sin x_i)^2, x_0=[0.2, 0.2, ...,0.2]
$$


```python
# make data(2 dimension)
x = sp.symbols("x1:3")
f = (2 - (sp.cos(x[0]) + sp.cos(x[1])) + (1 - sp.cos(x[0])) - sp.sin(x[0]))**2 + \
    (2 - (sp.cos(x[0]) + sp.cos(x[1])) + 2 * (1 - sp.cos(x[1])) - sp.sin(x[1]))**2
x_0 = (0.1, 0.1) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/e56a045605904c2b92331f9b967c1fbb.png#pic_center">

    


## Extended Rosenbrock function

$$
f(x)=\sum_{i=1}^{n/2}c(x_{2i}-x_{2i-1}^2)^2+(1-x_{2i-1})^2, x_0=[-1.2, 1, ...,-1.2, 1]. c=100
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = 100 * (x[1] - x[0]**2)**2 + \
    (1 - x[0])**2 + \
    100 * (x[3] - x[2]**2)**2 + \
    (1 - x[2])**2
x_0 = (-2, 2, -2, 2) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/73201d641ea941f7aeae65186085516d.png#pic_center">

    


## Generalized Rosenbrock function

$$
f(x)=\sum_{i=1}^{n-1}c(x_{i+1}-x_i^2)^2+(1-x_i)^2, x_0=[-1.2, 1, ...,-1.2, 1], c=100.
$$


```python
# make data(2 dimension)
x = sp.symbols("x1:3")
f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
x_0 = (-1, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/98ebf5d4110e4fefb7b46cadaf93c83a.png#pic_center">

    


## Extended White & Holst function

$$
f(x)=\sum_{i=1}^{n/2}c(x_{2i}-x_{2i-1}^3)^2+(1-x_{2i-1})^2, x_0=[-1.2, 1, ...,-1.2, 1]. c=100
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = 100 * (x[1] - x[0]**3)**2 + \
    (1 - x[0])**2 + \
    100 * (x[3] - x[2]**3)**2 + \
    (1 - x[2])**2
x_0 = (-1, 0.5, -1, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/e9c62804c55844278c23856f0903f9ab.png#pic_center">

    


## Extended Penalty function

$$
f(x)=\sum_{i=1}^{n-1} (x_i-1)^2+(\sum_{j=1}^{n}x_j^2-0.25)^2, x_0=[1,2,...,n].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (x[0] - 1)**2 + (x[1] - 1)**2 + (x[2] - 1)**2 + \
    ((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) - 0.25)**2
x_0 = (5, 5, 5, 5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/c4782f7e4ca64b2fabdd688e30f7dd74.png#pic_center">

    


## Perturbed Quadratic function

$$
f(x)=\sum_{i=1}^{n}ix_i^2+\frac{1}{100}(\sum_{i=1}^{n}x_i)^2, x_0=[0.5,0.5,...,0.5].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + 4*x[3]**2 + \
    0.01 * (x[0] + x[1] + x[2] + x[3])**2
x_0 = (1, 1, 1, 1) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/7bd48f86a045437c951e5cc0aafc5077.png#pic_center">

    


## Raydan 1 function

$$
f(x)=\sum_{i=1}^{n}\frac{i}{10}(\exp{x_i}-x_i), x_0=[1,1,...,1].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = 0.1 * (sp.exp(x[0]) - x[0]) + \
    0.2 * (sp.exp(x[1]) - x[1]) + \
    0.3 * (sp.exp(x[2]) - x[2]) + \
    0.4 * (sp.exp(x[3]) - x[3])
x_0 = (0.5, 0.5, 0.5, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/6eb63544330b48faa84acc3debe5adac.png#pic_center">

    


## Raydan 2 function

$$
f(x)=\sum_{i=1}^{n}(\exp{x_i}-x_i), x_0=[1,1,...,1].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (sp.exp(x[0]) - x[0]) + \
    (sp.exp(x[1]) - x[1]) + \
    (sp.exp(x[2]) - x[2]) + \
    (sp.exp(x[3]) - x[3])
x_0 = (2, 2, 2, 2) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/6bbde12aa7c94ddab87d9dd3f30c44a8.png#pic_center">

    


## Diagonal 1 function

$$
f(x)=\sum_{i=1}^{n}(\exp{x_i}-ix_i), x_0=[1/n,1/n,...,1/n].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (sp.exp(x[0]) - x[0]) + \
    (sp.exp(x[1]) - 2 * x[1]) + \
    (sp.exp(x[2]) - 3 * x[2]) + \
    (sp.exp(x[3]) - 4 * x[3])
x_0 = (0.5, 0.5, 0.5, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/fcc0addcace848a1988c2d480aa30b0d.png#pic_center">

    


## Diagonal 2 function

$$
f(x)=\sum_{i=1}^{n}(\exp{x_i}-\frac{x_i}{i}), x_0=[1/1,1/2,...,1/n].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (sp.exp(x[0]) - x[0]) + \
    (sp.exp(x[1]) - x[1] / 2) + \
    (sp.exp(x[2]) - x[2] / 3) + \
    (sp.exp(x[3]) - x[3] / 4)
x_0 = (0.9, 0.6, 0.4, 0.3) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/aeb5db56a88f45459430577b48dc66b8.png#pic_center">

    


## Diagonal 3 function

$$
f(x)=\sum_{i=1}^{n}(\exp{x_i}-i\sin(x_i)), x_0=[1,1,...,1].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (sp.exp(x[0]) - sp.sin(x[0])) + \
    (sp.exp(x[1]) - 2 * sp.sin(x[1])) + \
    (sp.exp(x[2]) - 3 * sp.sin(x[2])) + \
    (sp.exp(x[3]) - 4 * sp.sin(x[3]))
x_0 = (0.5, 0.5, 0.5, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/de9c8c8469684d7a9d97b18011522f31.png#pic_center">

    


## Hager function 

$$
f(x)=\sum_{i=1}^{n}(\exp{x_i}-\sqrt{i}x_i), x_0=[1,1,...,1].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = (sp.exp(x[0]) - x[0]) + \
    (sp.exp(x[1]) - sp.sqrt(2) * x[1]) + \
    (sp.exp(x[2]) - sp.sqrt(3) * x[2]) + \
    (sp.exp(x[3]) - sp.sqrt(4) * x[3])
x_0 = (0.5, 0.5, 0.5, 0.5) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/b9ba05071649471eac03da29a5c60b8d.png#pic_center">

    


## Generalized Tridiagonal 1 function

$$
f(x)=\sum_{i=1}^{n-1}(x_i+x_{i+1}-3)^2+(x_i-x_{i+1}+1)^4, x_0=[2,2,...,2].
$$


```python
# make data(3 dimension)
x = sp.symbols("x1:4")
f = (x[0] + x[1] - 3)**2 + (x[0] - x[1] + 1)**4 + \
    (x[1] + x[2] - 3)**2 + (x[1] - x[2] + 1)**4
x_0 = (1, 1, 1) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/3aa311c9527c490e923c6b8c51045e26.png#pic_center">

    


## Extended Tridiagonal 1 function:

$$
f(x)=\sum_{i=1}^{n/2}(x_{2i-1}+x_{2i}-3)^2+(x_{2i-1}-x_{2i}+1)^4, x_0=[2,2,...,2].
$$


```python
# make data(2 dimension)
x = sp.symbols("x1:3")
f = (x[0] + x[1] - 3)**2 + (x[0] - x[1] + 1)**4
x_0 = (1, 1) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    
<img src="https://img-blog.csdnimg.cn/001c0928546441d3a9412f7c58538bdf.png#pic_center">

    


## Extended TET function : (Three exponential terms)

$$
f(x)=\sum_{i=1}^{n/2}((\exp x_{2i-1} + 3x_{2i} - 0.1) + \exp (x_{2i-1} - 3x_{2i} - 0.1) + \exp (-x_{2i-1}-0.1)), x_0=[0.1,0.1,...,0.1].
$$


```python
# make data(4 dimension)
x = sp.symbols("x1:5")
f = sp.exp(x[0] + 3*x[1] - 0.1) + sp.exp(x[0] - 3*x[1] - 0.1) + sp.exp(-x[0] - 0.1) + \
    sp.exp(x[2] + 3*x[3] - 0.1) + sp.exp(x[2] - 3*x[3] - 0.1) + sp.exp(-x[2] - 0.1)
x_0 = (0.2, 0.2, 0.2, 0.2) # Random given
```


```python
# train
color, values, title = train(funcs=f, args=x, x_0=x_0)
```


```python
# test
test(color, values, title)
```


    

<img src="https://img-blog.csdnimg.cn/de5626fcd52f4d21ba24c6d751569811.png#pic_center">
