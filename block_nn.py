import optimtool.unconstrain as ou
from optimtool.base import sp, np

def _block_add(terms, block=32):
    cur = list(terms)
    while len(cur) > 1:
        cur = [sp.Add(*cur[i:i + block]) for i in range(0, len(cur), block)]
    return cur[0]

def gen_nn(X_data, y_data, hidden_dims=(), task='classification', epsilon=0.0, block=32):
    n_samples, n_features = X_data.shape
    if task == 'classification':
        if y_data.ndim == 1:
            classes = np.unique(y_data)
            n_outputs = classes.size
            y_processed = np.eye(n_outputs)[np.searchsorted(classes, y_data)]
        else:
            n_outputs = y_data.shape[1]
            y_processed = y_data.astype(float)
    elif task == 'regression':
        if y_data.ndim == 1:
            n_outputs = 1
            y_processed = y_data.reshape(-1, 1).astype(float)
        else:
            n_outputs = y_data.shape[1]
            y_processed = y_data.astype(float)
    else:
        raise ValueError(f"Support classification or regression. not support {task}.")
    layer_dims = [n_features] + list(hidden_dims) + [n_outputs]
    n_layers = len(layer_dims) - 1
    W_list, b_list, params = [], [], []
    for i in range(n_layers):
        prev_dim, curr_dim = layer_dims[i], layer_dims[i + 1]
        W_syms = sp.symbols(
            [f'W{i}_{j}_{k}' for j in range(curr_dim) for k in range(prev_dim)], real=True)
        b_syms = sp.symbols([f'b{i}_{j}' for j in range(curr_dim)], real=True)
        W_list.append([W_syms[j * prev_dim:(j + 1) * prev_dim] for j in range(curr_dim)])
        b_list.append(b_syms)
        params.extend(W_syms)
        params.extend(b_syms)
    H = X_data.tolist()                                  # (n, in_dim)
    for i in range(n_layers):
        in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
        Wi, bi = W_list[i], b_list[i]
        H = [[sp.Add(*[H[s][k] * Wi[j][k] for k in range(in_dim)], bi[j])
              for j in range(out_dim)]
             for s in range(n_samples)]
    logits = H                                           # (n, n_outputs)
    eps = 0.0 if epsilon is None else epsilon
    terms = []
    if task == 'classification':
        # -Σ_c y_c log softmax_c = logΣ_c exp(logit_c) - Σ_c y_c·logit_c
        for s in range(n_samples):
            l = logits[s]
            lse = sp.log(sp.Add(*[sp.exp(v) for v in l]) + eps)
            lin = []
            for c in range(n_outputs):
                yv = y_processed[s, c]
                if yv == 0:
                    continue
                lin.append(l[c] if yv == 1 else yv * l[c])
            terms.append(lse - sp.Add(*lin))
        denom = n_samples
    else:
        for s in range(n_samples):
            for o in range(n_outputs):
                d = sp.Add(logits[s][o], -float(y_processed[s, o]))
                terms.append(sp.Mul(d, d))
        denom = n_samples * n_outputs
    f_sym = _block_add(terms, block) / denom
    return f_sym, params

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=64, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=0)
hidden = [2,]
f_sym, params = gen_nn(X, y, hidden_dims=hidden)
def xavier_normal(layer_dims):
    theta = []
    for i in range(len(layer_dims) - 1):
        fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        theta.extend(np.random.randn(fan_in * fan_out) * std)
        theta.extend(np.zeros(fan_out))
    return theta
layer_dims = [X.shape[1]] + hidden + [len(np.unique(y))]
theta0 = xavier_normal(layer_dims)
ou.gradient_descent.barzilar_borwein(f_sym, params, theta0, verbose=True, epsilon=1e-2)