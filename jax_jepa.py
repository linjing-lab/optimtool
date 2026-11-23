import jax
import jax.numpy as jnp
from collections import namedtuple
from jax import lax
from jax_nn import xavier_normal

jax.config.update("jax_enable_x64", True)
def make_bfgs(fg, hess_fn, n, max_iter=60, max_step=0.5, damp=1e-2,
              c1=0.3, c2=0.5, ls_eps=1e-3):
    I = jnp.eye(n)

    def h2h_jit(hessian):
        def cond(H):
            return jnp.linalg.matrix_rank(H) < n
        return lax.while_loop(cond, lambda H: H + I, hessian)

    def wolfe(theta, dk, f0, g0, Xb, theta_tgt):
        dt = dk.dtype
        slope = jnp.dot(g0, dk)

        def cond(c):
            lo, hi, alpha, done = c
            return (~done) & (hi - lo > ls_eps) & (lo < hi)

        def body(c):
            lo, hi, alpha, _ = c
            f1, g1 = fg(theta + alpha * dk, Xb, theta_tgt)
            armijo = f1 <= f0 + c1 * alpha * slope
            curvature = jnp.dot(g1, dk) >= c2 * slope
            done = armijo & curvature
            lo2 = jnp.where(armijo, alpha, lo)
            hi2 = jnp.where(armijo, hi, alpha)
            a2 = 0.5 * (lo2 + hi2)
            return (lo2, hi2, jnp.where(done, alpha, a2), done)

        init = (jnp.array(0., dt), jnp.array(2., dt),
                jnp.array(1., dt), jnp.array(False))
        return lax.while_loop(cond, body, init)[2]

    @jax.jit
    def solve(theta, Xb, theta_tgt, epsilon):
        f0, g0 = fg(theta, Xb, theta_tgt)
        H = h2h_jit(hess_fn(theta, Xb, theta_tgt))
        H = jnp.where(jnp.all(jnp.isfinite(H)), H, I)
        Hd = H + damp * I
        dk = -jnp.linalg.solve(Hd, g0)
        dk = jnp.clip(dk, -max_step, max_step)
        dk = jnp.where(jnp.dot(g0, dk) >= 0,
                       -jnp.clip(g0, -max_step, max_step), dk)

        def cond(c):
            _, _, d, _, _, it = c
            return (it < max_iter) & (jnp.linalg.norm(d) >= epsilon)

        def body(c):
            th, H, d, f, g, it = c
            alpha = wolfe(th, d, f, g, Xb, theta_tgt)
            s = alpha * d
            th2 = th + s
            f1, g1 = fg(th2, Xb, theta_tgt)
            yk = g1 - g
            sTy = jnp.dot(s, yk)
            Hs = H @ s
            sTHs = jnp.dot(s, Hs)
            ok = (jnp.all(jnp.isfinite(yk)) & jnp.isfinite(sTy) & (sTy > 1e-12)
                  & jnp.all(jnp.isfinite(Hs)) & jnp.isfinite(sTHs)
                  & (sTHs > 1e-12))

            safe1 = jnp.where(sTy > 1e-12, sTy, 1.0)
            safe2 = jnp.where(sTHs > 1e-12, sTHs, 1.0)
            H2 = jnp.where(ok,
                           H + jnp.outer(yk, yk) / safe1
                             - jnp.outer(Hs, Hs) / safe2,
                           I)
            H2 = jnp.where(jnp.all(jnp.isfinite(H2)), H2, I)
            d2 = -jnp.linalg.solve(H2 + damp * I, g1)
            d2 = jnp.clip(d2, -max_step, max_step)
            d2 = jnp.where(jnp.dot(g1, d2) >= 0,
                           -jnp.clip(g1, -max_step, max_step), d2)
            return (th2, H2, d2, f1, g1, it + 1)

        carry = (theta, H, dk, f0, g0, jnp.array(0, jnp.int32))
        out = lax.while_loop(cond, body, carry)
        return out[0], f0, out[3]

    return solve

def make_mlp(layer_dims, key, dtype):
    layer_dims = [int(d) for d in layer_dims]
    sizes = list(zip(layer_dims[:-1], layer_dims[1:]))
    init = xavier_normal(layer_dims, key, dtype=dtype)

    cuts, acc = [], 0
    for prev_dim, curr_dim in sizes:
        acc += prev_dim * curr_dim
        cuts.append(acc)
        acc += curr_dim
        cuts.append(acc)
    boundaries = cuts[:-1]

    def apply(theta, x):
        parts = jnp.split(theta.reshape(-1), boundaries)
        Ws = [parts[2 * i].reshape(curr_dim, prev_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        bs = [parts[2 * i + 1].reshape(curr_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        H = x
        for i, (W, b) in enumerate(zip(Ws, bs)):
            H = H @ W.T + b
            if i < len(Ws) - 1:
                H = jnp.tanh(H)
        return H

    return init, apply, int(init.size)


JEPA = namedtuple('JEPA',
                  'f_fun make_f_fun stats theta0 theta_tgt0 n_enc encode '
                  'momentum var_weight ctx_steps tgt_steps repr_dim '
                  'f fg bfgs evaluate')

def gen_jepa(X_data, ctx_steps=8, tgt_steps=3, repr_dim=4,
             enc_hidden=8, pred_hidden=12,
             var_weight=0.1, momentum=0.9, key=None, seed=0):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X_data = jnp.asarray(X_data)
    dtype = jnp.result_type(X_data.dtype, jnp.float32)
    X_data = X_data.astype(dtype)

    if X_data.ndim != 3:
        raise ValueError("JEPA expects X_data with shape (n, L, d).")
    if X_data.shape[1] < ctx_steps + tgt_steps:
        raise ValueError("sequence length must be >= ctx_steps + tgt_steps.")

    d = int(X_data.shape[2])
    enc_dims = [d, enc_hidden, repr_dim]
    pred_dims = [repr_dim, pred_hidden, tgt_steps * repr_dim]
    k1, k2 = jax.random.split(key)
    th_enc, enc_apply, n_enc = make_mlp(enc_dims, k1, dtype)
    th_pred, pred_apply, _ = make_mlp(pred_dims, k2, dtype)
    theta0 = jnp.concatenate([th_enc, th_pred])
    theta_tgt0 = jnp.array(th_enc)

    def encode(theta_enc, frames):
        return enc_apply(theta_enc, frames)

    def _stats(theta, Xb, theta_tgt):
        ctx = Xb[:, :ctx_steps, :]
        tgt = Xb[:, ctx_steps:ctx_steps + tgt_steps, :]

        z_ctx = enc_apply(theta[:n_enc], ctx)                 # (m, T, r)
        z_pool = jnp.mean(z_ctx, axis=1)                      # (m, r)
        z_pred = pred_apply(theta[n_enc:], z_pool).reshape(-1, tgt_steps, repr_dim)
        z_tgt = jax.lax.stop_gradient(enc_apply(theta_tgt, tgt))

        mse = jnp.mean((z_pred - z_tgt) ** 2)
        flat = z_pred.reshape(-1, repr_dim)
        std = jnp.sqrt(jnp.var(flat, axis=0) + 1e-4)
        var_reg = jnp.mean((1.0 - std) ** 2)
        return mse, var_reg, std

    def _loss(theta, Xb, theta_tgt):
        mse, var_reg, _ = _stats(theta, Xb, theta_tgt)
        return mse + var_weight * var_reg

    f_jit = jax.jit(_loss)
    fg_jit = jax.jit(jax.value_and_grad(_loss))
    hess_jit = jax.jit(jax.hessian(_loss))
    stats_jit = jax.jit(_stats)

    @jax.jit
    def evaluate(theta, Xb, X_full, theta_tgt):
        mse, _, std = stats_jit(theta, Xb, theta_tgt)
        return mse, jnp.mean(std), f_jit(theta, X_full, theta_tgt)

    def f_fun(theta, theta_tgt):
        return f_jit(theta, X_data, theta_tgt)

    def make_f_fun(Xb, theta_tgt):
        Xb = jnp.asarray(Xb).astype(dtype)
        return lambda theta: f_jit(theta, Xb, theta_tgt)

    return JEPA(f_fun=f_fun, make_f_fun=make_f_fun, stats=stats_jit,
                theta0=theta0, theta_tgt0=theta_tgt0, n_enc=n_enc,
                encode=encode, momentum=momentum, var_weight=var_weight,
                ctx_steps=ctx_steps, tgt_steps=tgt_steps, repr_dim=repr_dim,
                f=f_jit, fg=fg_jit,
                bfgs=make_bfgs(fg_jit, hess_jit, int(theta0.size)),
                evaluate=evaluate)

def train_jepa(model, theta, theta_tgt, X,
               epochs=3, batch_size=64, epsilon=5e-2,
               seed=0, verbose=True, patience=10,
               min_delta=1e-8, monitor='batch'):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")
    X = jnp.asarray(X)
    n = X.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    eps = jnp.asarray(epsilon)
    history = []
    early_stopped = False

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated, no_decrease, prev_full = 0, 0, float('inf')

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            Xb = X[idx]

            theta, l0, l1 = model.bfgs(theta, Xb, theta_tgt, eps)
            mse, stdm, l_full = model.evaluate(theta, Xb, X, theta_tgt)
            l0, l1, mse, stdm, l_full = map(
                float, jax.device_get((l0, l1, mse, stdm, l_full)))

            theta_tgt = (model.momentum * theta_tgt
                         + (1.0 - model.momentum) * theta[:model.n_enc])

            decreased = (l_full < prev_full - min_delta) if monitor == 'full' \
                        else (l1 < l0 - min_delta)
            if monitor == 'full':
                prev_full = l_full
            no_decrease = 0 if decreased else no_decrease + 1
            if abs(l1 - l0) > 1e-12:
                n_updated += 1

            history.append(dict(epoch=epoch, batch=b, n=int(idx.shape[0]),
                                loss_before=l0, loss_after=l1,
                                loss_full=l_full, mse=mse, std=stdm))
            if verbose:
                print('epoch {} | batch {:>2d} | n={:<3d} | jepa loss '
                      '{:.6f} -> {:.6f} | full {:.6f} | pred mse {:.6f} '
                      '| repr std {:.4f}'
                      .format(epoch, b, int(idx.shape[0]), l0, l1, l_full,
                              mse, stdm))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d} | {} consecutive '
                          'batches without loss decrease\n'
                          .format(epoch, b, patience))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full loss {:.6f} | updated {}/{} batches\n'
                  .format(epoch, l_full, n_updated, n_batches))
        if early_stopped:
            break

    return theta, theta_tgt, history

def make_series(n_windows=512, ctx_steps=8, tgt_steps=3, d=4, seed=7):
    key = jax.random.PRNGKey(seed)
    k_phase, k_mix, k_noise = jax.random.split(key, 3)
    total = n_windows + ctx_steps + tgt_steps
    t = jnp.linspace(0, 60, total)[:, None]
    phases = jax.random.uniform(k_phase, (1, d), minval=0., maxval=2 * jnp.pi,
                                dtype=jnp.float64)
    freqs = jnp.linspace(0.35, 0.95, d)[None, :]
    base = (jnp.sin(freqs * t + phases)
            + 0.35 * jnp.sin(2.3 * freqs * t + 0.7 * phases))
    mix = jax.random.normal(k_mix, (d, d), dtype=jnp.float64) / jnp.sqrt(d)
    noise = 0.02 * jax.random.normal(k_noise, base.shape, dtype=jnp.float64)
    signal = base @ mix + noise
    L = ctx_steps + tgt_steps
    starts = jnp.arange(n_windows)
    return jax.vmap(lambda i: jax.lax.dynamic_slice(signal, (i, 0), (L, d)))(starts)

def linear_probe(model, theta, theta0, X):
    ctx = X[:, :model.ctx_steps, :]
    y = X[:, model.ctx_steps:model.ctx_steps + model.tgt_steps, :]
    y = y.reshape(X.shape[0], -1)

    def r2(A):
        A = jnp.concatenate([A, jnp.ones((A.shape[0], 1))], axis=1)
        W = jnp.linalg.solve(A.T @ A + 1e-3 * jnp.eye(A.shape[1]), A.T @ y)
        pred = A @ W
        return 1.0 - jnp.sum((y - pred) ** 2) / jnp.sum((y - y.mean(axis=0)) ** 2)

    Z = jnp.mean(model.encode(theta[:model.n_enc], ctx), axis=1)
    Z0 = jnp.mean(model.encode(theta0[:model.n_enc], ctx), axis=1)
    raw = X[:, model.ctx_steps - 1, :]
    return float(r2(Z)), float(r2(Z0)), float(r2(raw))


if __name__ == '__main__':
    SEED, EPOCHS, BATCH_SIZE, EPSILON = 42, 3, 64, 5e-2
    CTX, TGT, D = 8, 3, 4

    X = make_series(n_windows=320, ctx_steps=CTX, tgt_steps=TGT, d=D, seed=7)
    X = (X - X.mean(axis=(0, 1))) / X.std(axis=(0, 1))
    print('data shape (n, L, d):', X.shape)

    model = gen_jepa(X, ctx_steps=CTX, tgt_steps=TGT, repr_dim=4,
                     enc_hidden=8, pred_hidden=12,
                     var_weight=0.1, momentum=0.99, seed=SEED)
    theta, theta_tgt = model.theta0, model.theta_tgt0
    print('online params: {} (encoder {} + predictor {}) | target params: {}'
          .format(theta.size, model.n_enc, theta.size - model.n_enc,
                  theta_tgt.size))

    f0_init, g_init = model.fg(theta, jnp.asarray(X[:BATCH_SIZE]), theta_tgt)
    print('init full loss: {:.6f} | grad shape: {}'
          .format(float(model.f_fun(theta, theta_tgt)), g_init.shape))

    theta, theta_tgt, hist = train_jepa(
        model, theta, theta_tgt, X,
        epochs=EPOCHS, batch_size=BATCH_SIZE, epsilon=EPSILON,
        seed=SEED, verbose=True, patience=10, monitor='batch')

    print('final full loss: {:.6f}'.format(float(model.f_fun(theta, theta_tgt))))
    per_epoch = {}
    for h in hist:
        per_epoch[h['epoch']] = h['loss_full']
    print('epoch-end full loss:',
          [round(per_epoch[e], 6) for e in sorted(per_epoch)])

    r2_repr, r2_rand, r2_raw = linear_probe(model, theta, model.theta0, X)
    print('linear probe R^2 (predict next {} frames) | trained repr: {:.4f} '
          '| random-init repr: {:.4f} | raw last frame: {:.4f}'
          .format(TGT, r2_repr, r2_rand, r2_raw))
