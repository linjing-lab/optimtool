import sympy as sp
import matplotlib.pyplot as plt
import optimtool as oo

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

x = sp.symbols("x1:5")
f = (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + \
    (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + \
    (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + \
    (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2
x_0 = (1, -1, 1, -1) # Random given

# train
color, values, title = train(funcs=f, args=x, x_0=x_0)

# test
test(color, values, title)