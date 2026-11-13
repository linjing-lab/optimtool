import jax
import jax.numpy as jnp
from typing import Callable, Iterable

def wolfe(fun: Callable,
          res0, 
          x_0: Iterable, 
          dk, 
          c1: float=0.3, 
          c2: float=0.5, 
          alphas: float=0., 
          alphae: float=2.,
          eps: float=1e-3) -> float:
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas >= 0
    assert alphas < alphae
    assert eps > 0
    alpha = 1
    f0 = fun(x_0)
    while alphas < alphae:
        x = x_0 + (alpha*dk)[0]
        f1 = fun(x)
        if f1 <= f0 + c1*alpha*res0.dot(dk.T):
            res1 = jax.grad(fun)(x)
            if res1.dot(dk.T) >= c2*res0.dot(dk.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if jax.numpy.abs(alphas - alphae) < eps:
            break
    return alpha

def h2h(hessian):
    l = hessian.shape[0] # hessian.shape = (l, l)
    while 1:
        rank = jnp.linalg.matrix_rank(hessian)
        if rank == l:
            break
        else:
            hessian += jnp.identity(l)
    return hessian

def bfgs(fun: Callable, 
         x_0: Iterable,
         verbose: bool=False,
         epsilon: float=1e-6, 
         k: int=0):
    hessian = jax.hessian(fun)(x_0)
    hessian = h2h(hessian)
    while 1:
        gradient = jax.grad(fun)(x_0)
        dk = -jax.numpy.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if verbose:
            print('{}\t{}\t{}'.format(x_0, fun(x_0), k))
        if jax.numpy.linalg.norm(dk) >= epsilon:
            alpha = wolfe(fun, gradient, x_0, dk)
            delta = alpha * dk # sk, same as learning_date * grad(corrected by hessian and jax.grad)
            x_0 += delta[0]
            yk = jax.grad(fun)(x_0) - gradient
            s = delta.reshape(-1)                    
            sTy = float(s.dot(yk))                   
            if jnp.all(jnp.isfinite(yk)) and jnp.isfinite(sTy) and sTy > 1e-12:
                Hs = hessian.dot(delta.T)
                sTHs = float(s.dot(Hs.reshape(-1)))
                if jnp.isfinite(sTHs) and sTHs > 1e-12:
                    hessian += jnp.outer(yk, yk) / sTy - Hs.dot(Hs.T) / sTHs
            else:
                hessian = h2h(jnp.eye(hessian.shape[0], dtype=hessian.dtype))
            k += 1
        else:
            break
    return x_0

def xavier_normal(layer_dims, key, dtype=jnp.float64):
    layer_dims = [int(d) for d in layer_dims]
    chunks = []
    for i, (fan_in, fan_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        sub = jax.random.fold_in(key, i)
        chunks.append(jax.random.normal(sub, (fan_in * fan_out,), dtype=dtype) * std)
        chunks.append(jnp.zeros((fan_out,), dtype=dtype))
    return jnp.concatenate(chunks)


def gen_nn(X_data, y_data, hidden_dims=(), task='classification', key=None, seed=0):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X_data = jnp.asarray(X_data)
    y_data = jnp.asarray(y_data)
    dtype = jnp.result_type(X_data.dtype, jnp.float32)
    X_data = X_data.astype(dtype)

    n_samples, n_features = X_data.shape
    n_features = int(n_features)

    if task == 'classification':
        if y_data.ndim == 1:
            classes = jnp.unique(y_data)
            n_outputs = int(classes.size)
            y_processed = jnp.eye(n_outputs, dtype=dtype)[jnp.searchsorted(classes, y_data)]
        else:
            n_outputs = int(y_data.shape[1])
            y_processed = y_data.astype(dtype)
    elif task == 'regression':
        if y_data.ndim == 1:
            n_outputs = 1
            y_processed = y_data.reshape(-1, 1).astype(dtype)
        else:
            n_outputs = int(y_data.shape[1])
            y_processed = y_data.astype(dtype)
    else:
        raise ValueError(f"Support classification or regression. not support {task}.")

    layer_dims = jnp.concatenate([
        jnp.asarray([n_features]),
        jnp.asarray(hidden_dims, dtype=jnp.int32).reshape(-1),
        jnp.asarray([n_outputs]),
    ])
    layer_dims = [int(d) for d in layer_dims]
    sizes = list(zip(layer_dims[:-1], layer_dims[1:]))

    theta0 = xavier_normal(layer_dims, key, dtype=dtype)

    cuts, acc = [], 0
    for prev_dim, curr_dim in sizes:
        acc += prev_dim * curr_dim
        cuts.append(acc)
        acc += curr_dim
        cuts.append(acc)
    boundaries = cuts[:-1]

    def unpack(theta):
        parts = jnp.split(theta.reshape(-1), boundaries)
        Ws = [parts[2 * i].reshape(curr_dim, prev_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        bs = [parts[2 * i + 1].reshape(curr_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        return Ws, bs

    def forward(theta):
        Ws, bs = unpack(theta)
        H = X_data
        for i, (W, b) in enumerate(zip(Ws, bs)):
            H = H @ W.T + b
        return H

    def f_fun(theta):
        logits = forward(theta)
        if task == 'classification':
            lse = jax.nn.logsumexp(logits, axis=1)
            return jnp.mean(lse - jnp.sum(y_processed * logits, axis=1))
        else:
            d = logits - y_processed
            return jnp.sum(d * d) / (n_samples * n_outputs)

    return f_fun, theta0

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_breast_cancer

    jax.config.update("jax_enable_x64", True)

    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    hidden = [4,]
    print(X.shape, y.shape)

    f_fun, theta0 = gen_nn(X, y, hidden_dims=hidden, task="classification", seed=42)

    f_jit = jax.jit(f_fun)
    print(f_jit(theta0), jax.grad(f_jit)(theta0).shape)

    theta_opt = bfgs(f_fun, theta0, verbose=True, epsilon=1e-2)
    print(f_fun(theta_opt))