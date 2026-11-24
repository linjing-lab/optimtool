import jax
import jax.numpy as jnp
from collections import namedtuple
from jax_jepa import make_bfgs, make_mlp, make_series

jax.config.update("jax_enable_x64", True)

VAE = namedtuple('VAE',
                 'theta0 n_enc n_params latent_dim obs_dim L D beta eps_all '
                 'parts loss loss_full posterior encode_mu decode generate '
                 'bfgs evaluate')

def gen_vae(X, latent_dim=4, hidden=5, beta=1.0,
            max_iter=6, max_step=0.05, damp=0.1,
            key=None, seed=0, eps_seed=None):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X = jnp.asarray(X)
    dtype = jnp.result_type(X.dtype, jnp.float32)
    X = X.astype(dtype)
    if X.ndim != 3:
        raise ValueError("expects (n, L, d)")
    n, L, d = X.shape
    obs_dim = int(L * d)
    X_flat = X.reshape(n, obs_dim)
    enc_dims = [obs_dim, hidden, 2 * latent_dim]
    dec_dims = [latent_dim, hidden, obs_dim]

    k_enc, k_dec, k_eps = jax.random.split(key, 3)
    if eps_seed is not None:
        k_eps = jax.random.PRNGKey(int(eps_seed))

    th_enc, enc_apply, n_enc = make_mlp(enc_dims, k_enc, dtype)
    th_dec, dec_apply, _ = make_mlp(dec_dims, k_dec, dtype)
    theta0 = jnp.concatenate([th_enc, th_dec])
    n_params = int(theta0.size)
    eps_all = jax.random.normal(k_eps, (n, latent_dim), dtype=dtype)

    def infer(theta, xb, eps):
        out = enc_apply(theta[:n_enc], xb)
        mu, raw_lv = jnp.split(out, 2, axis=-1)
        logvar = 3.0 * jnp.tanh(raw_lv / 3.0)
        z = mu + jnp.exp(0.5 * logvar) * eps
        return z, mu, logvar

    def posterior(theta, xb):
        mu, raw_lv = jnp.split(enc_apply(theta[:n_enc], xb), 2, axis=-1)
        return mu, 3.0 * jnp.tanh(raw_lv / 3.0)

    def encode_mu(theta, xb):
        return posterior(theta, xb)[0]

    def decode(theta, z):
        return dec_apply(theta[n_enc:], z)

    def parts(theta, xb, eps):
        z, mu, logvar = infer(theta, xb, eps)
        xhat = dec_apply(theta[n_enc:], z)
        recon = 0.5 * jnp.mean(jnp.sum((xb - xhat) ** 2, axis=-1))
        kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar),
                                     axis=-1))
        return recon, kl

    def loss(theta, xb, eps):
        recon, kl = parts(theta, xb, eps)
        return recon + beta * kl

    def loss_full(theta, xb, eps=None):
        return loss(theta, xb, eps_all if eps is None else eps)

    fg = jax.jit(jax.value_and_grad(loss))
    hess_fn = jax.jit(jax.hessian(loss))

    def safe_hess(theta, xb, eps):
        H = hess_fn(theta, xb, eps)
        return jnp.where(jnp.isfinite(H), H, 0.0)

    bfgs = make_bfgs(fg, safe_hess, n_params, max_iter=max_iter,
                     max_step=max_step, damp=damp)

    @jax.jit
    def evaluate(theta, xb, x_full, eps_b):
        rec, kl = parts(theta, xb, eps_b)
        return rec, kl, loss(theta, xb, eps_b), loss_full(theta, x_full)

    return VAE(theta0=theta0, n_enc=n_enc, n_params=n_params,
               latent_dim=int(latent_dim), obs_dim=obs_dim, L=int(L), D=int(d),
               beta=beta, eps_all=eps_all,
               parts=parts, loss=loss, loss_full=loss_full,
               posterior=posterior, encode_mu=encode_mu, decode=decode,
               generate=lambda theta, z: dec_apply(theta[n_enc:], z),
               bfgs=bfgs, evaluate=evaluate), X_flat


def train_vae(model, theta, X_flat, epochs=3, batch_size=64,
              bfgs_epsilon=1e-1, seed=0, verbose=True,
              patience=0, min_delta=1e-8, monitor='full',
              resample_eps=True, return_best=True):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")
    n = X_flat.shape[0]
    if n % batch_size:
        raise ValueError("n_windows must be divisible by batch_size "
                         "to avoid recompiling make_bfgs on a shorter batch.")
    n_batches = n // batch_size
    eps_bfgs = jnp.asarray(bfgs_epsilon)
    eps_all = model.eps_all
    dtype = eps_all.dtype
    history = []
    early_stopped = False
    eps_key = jax.random.PRNGKey(seed + 10007)

    best_theta, best_full = theta, float('inf')
    prev_full, no_decrease = float('inf'), 0

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated = 0

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            Xb = X_flat[idx]
            if resample_eps:
                eps_key, k_step = jax.random.split(eps_key)
                eps_b = jax.random.normal(k_step,
                                          (batch_size, model.latent_dim),
                                          dtype=dtype)
            else:
                eps_b = eps_all[idx]

            theta, l0, l1 = model.bfgs(theta, Xb, eps_b, eps_bfgs)
            rec, kl, _, l_full = model.evaluate(theta, Xb, X_flat, eps_b)
            l0, l1, rec, kl, l_full = map(
                float, jax.device_get((l0, l1, rec, kl, l_full)))

            decreased = (l_full < prev_full - min_delta) if monitor == 'full' \
                        else (l1 < l0 - min_delta)
            if monitor == 'full':
                prev_full = l_full
            no_decrease = 0 if decreased else no_decrease + 1
            if abs(l1 - l0) > 1e-12:
                n_updated += 1
            if l_full < best_full - min_delta:
                best_full, best_theta = l_full, theta

            history.append(dict(epoch=epoch, batch=b, n=int(idx.shape[0]),
                                loss_before=l0, loss_after=l1,
                                loss_full=l_full, recon=rec, kl=kl,
                                best_full=best_full))
            if verbose:
                print('epoch {} | batch {:>2d} | n={:<3d} | -ELBO '
                      '{:.4f} -> {:.4f} | full {:.4f} | best {:.4f} '
                      '| recon {:.4f} | KL {:.4f}'
                      .format(epoch, b, int(idx.shape[0]), l0, l1, l_full,
                              best_full, rec, kl))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d}\n'.format(epoch, b))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full -ELBO {:.4f} | best {:.4f} | '
                  'updated {}/{} batches\n'
                  .format(epoch, l_full, best_full, n_updated, n_batches))
        if early_stopped:
            break

    return (best_theta if return_best else theta), history


if __name__ == '__main__':
    SEED, LATENT, HIDDEN, BETA = 42, 4, 5, 1.0
    CTX, TGT, D, N_WINDOWS = 4, 2, 4, 512
    BATCH_SIZE, EPOCHS = 64, 8
    EPS_SEED = 0

    X = make_series(n_windows=N_WINDOWS, ctx_steps=CTX, tgt_steps=TGT,
                    d=D, seed=7)
    X = (X - X.mean(axis=(0, 1))) / X.std(axis=(0, 1))
    print('data shape (n, L, d):', X.shape, '| L =', CTX + TGT)

    model, X_flat = gen_vae(X, latent_dim=LATENT, hidden=HIDDEN, beta=BETA,
                            max_iter=6, max_step=0.05, damp=0.1,
                            seed=SEED, eps_seed=EPS_SEED)
    print('params       : {} (encoder {} + decoder {}) | Hessian {}x{} | latent {}'
          .format(model.n_params, model.n_enc,
                  model.n_params - model.n_enc,
                  model.n_params, model.n_params, LATENT))
    theta = model.theta0
    print('init full -ELBO: {:.4f}'
          .format(float(model.loss_full(theta, X_flat))))

    theta, hist = train_vae(model, theta, X_flat, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, bfgs_epsilon=1e-1,
                            seed=SEED, verbose=True, patience=0,
                            monitor='full', resample_eps=True,
                            return_best=True)
    print('final full -ELBO: {:.4f}'
          .format(float(model.loss_full(theta, X_flat))))
    th = jnp.asarray(theta)
    mu = model.encode_mu(th, X_flat)
    rec_flat = model.decode(th, mu)
    print('reconstruction MSE (z-score space): {:.4f}'
          .format(float(jnp.mean((X_flat - rec_flat) ** 2))))

    fulls = [h['loss_full'] for h in hist]
    rises = sum(1 for i in range(1, len(fulls))
                if fulls[i] > fulls[i - 1] + 1e-8)
    max_rise = max((fulls[i] - fulls[i - 1]
                    for i in range(1, len(fulls))), default=0.0)
    print('stability     : best {:.4f} | end {:.4f} | rises {}/{} | '
          'max single-step rise {:.4f}'
          .format(min(fulls), fulls[-1], rises, len(fulls) - 1, max_rise))
