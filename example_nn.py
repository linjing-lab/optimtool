import optimtool.unconstrain as ou
from optimtool.base import sp, np

def gen_nn(X_data, y_data, hidden_dims=[], task='classification'):
    n_samples, n_features = X_data.shape
    if task == 'classification':
        if len(y_data.shape) == 1:
            n_outputs = len(np.unique(y_data))
            y_one_hot = np.zeros((n_samples, n_outputs))
            for i, label in enumerate(y_data):
                y_one_hot[i, int(label)] = 1
        else:
            n_outputs = y_data.shape[1]
            y_one_hot = y_data
        y_processed = y_one_hot
    elif task == 'regression':
        if len(y_data.shape) == 1:
            n_outputs = 1
            y_processed = y_data.reshape(-1, 1)
        else:
            n_outputs = y_data.shape[1]
            y_processed = y_data
    else:
        raise ValueError(f"Support classification or regression. not support {task}.")
    params = []
    layer_dims = [n_features] + hidden_dims + [n_outputs]
    for i in range(len(layer_dims)-1):
        prev_dim = layer_dims[i]
        curr_dim = layer_dims[i+1]
        for j in range(curr_dim):
            for k in range(prev_dim):
                params.append(sp.symbols(f'W{i}_{j}_{k}', real=True))
        for j in range(curr_dim):
            params.append(sp.symbols(f'b{i}_{j}', real=True))
    def forward(X_vec):
        idx = 0
        x = X_vec.copy()
        current_dim = n_features
        for i, h_dim in enumerate(hidden_dims):
            W_mat = []
            for j in range(h_dim):
                row = []
                for k in range(current_dim):
                    row.append(params[idx])
                    idx += 1
                W_mat.append(row)
            b_vec = []
            for j in range(h_dim):
                b_vec.append(params[idx])
                idx += 1
            z = []
            for j in range(h_dim):
                sum_val = 0
                for k in range(current_dim):
                    sum_val += W_mat[j][k] * x[k]
                sum_val += b_vec[j]
                z.append(sum_val)# sp.Max(0, sum_val)
            x = z
            current_dim = h_dim
        output_dim = n_outputs
        W_mat = []
        for j in range(output_dim):
            row = []
            for k in range(current_dim):
                row.append(params[idx])
                idx += 1
            W_mat.append(row)
        b_vec = []
        for j in range(output_dim):
            b_vec.append(params[idx])
            idx += 1
        out = []
        for j in range(output_dim):
            sum_val = 0
            for k in range(current_dim):
                sum_val += W_mat[j][k] * x[k]
            sum_val += b_vec[j]
            out.append(sum_val)
        return out
    f_sym = 0
    epsilon = 1e-10
    if task == 'classification':
        for s in range(n_samples):
            logits = forward(X_data[s].tolist())
            y_true = y_processed[s].tolist()
            log_sum_exp = sp.log(sum(sp.exp(l) for l in logits) + epsilon)
            probs = [sp.exp(l - log_sum_exp) for l in logits]
            for c in range(n_outputs):
                f_sym += -y_true[c] * sp.log(probs[c] + epsilon) # Softmax
        f_sym /= n_samples 
    else:  # regression
        predictions = []
        for s in range(n_samples):
            pred = forward(X_data[s].tolist())
            predictions.append(pred)
        for s in range(n_samples):
            for o in range(n_outputs):
                diff = predictions[s][o] - y_processed[s, o]
                f_sym += diff ** 2 # MSE
        f_sym /= (n_samples * n_outputs)
    return f_sym, params

# # classification
# X = np.array([
#     [2.0, 1.0, 0.5, 0.1],
#     [5.1, 3.1, 2.1, 1.05],
#     [5.0, 3.0, 2.0, 1.0],
#     [2.1, 0.9, 0.55, 0.15],
#     [8.0, 5.0, 4.0, 2.0],
#     [8.2, 5.2, 4.2, 2.1],
# ], dtype=np.float64)
# y = np.array([0, 1, 1, 0, 2, 2], dtype=np.int64)
# f_sym, params = gen_nn(X, y, hidden_dims=[2,])
# ou.gradient_descent.barzilar_borwein(f_sym, params, np.ones(len(params)).tolist(), verbose=True, epsilon=1e-2)

# regression
def gen_reg(n_samples=20, seed=0):
    np.random.seed(seed)
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    noise = 0.2 * np.random.randn(n_samples, 3)
    y1 = np.sin(X)
    y2 = 0.5 * X
    y3 = np.cos(X) + 0.2 * X
    y = np.hstack([y1, y2, y3]) + noise
    return X, y

X, y = gen_reg(n_samples=5)
f_sym, params = gen_nn(X, y, hidden_dims=[2,], task='regression')
ou.newton_quasi.bfgs(f_sym, params, np.ones(len(params)).tolist(), verbose=True, epsilon=1e-4)