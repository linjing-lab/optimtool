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
            max_iter=40, key=None, seed=0):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 3:
        raise ValueError("expects (n, L, d)")
    n, L, d = X.shape
    obs_dim = int(L * d)
    X_flat = X.reshape(n, obs_dim)
    enc_dims = [obs_dim, hidden, 2 * latent_dim]
    dec_dims = [latent_dim, hidden, obs_dim]
    k1, k2 = jax.random.split(key)
    th_enc, enc_apply, n_enc = make_mlp(enc_dims, k1, jnp.float64)
    th_dec, dec_apply, _ = make_mlp(dec_dims, k2, jnp.float64)
    theta0 = jnp.concatenate([th_enc, th_dec])
    n_params = int(theta0.size)
    eps_all = jax.random.normal(jax.random.fold_in(key, 11),
                                (n, latent_dim), dtype=jnp.float64)

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

    def loss_full(theta, xb):
        return loss(theta, xb, eps_all)

    fg = jax.jit(jax.value_and_grad(loss))
    hess_fn = jax.jit(jax.hessian(loss))
    bfgs = make_bfgs(fg, hess_fn, n_params, max_iter=max_iter,
                     max_step=0.5, damp=1e-2)

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
              patience=0, min_delta=1e-8, monitor='batch'):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")
    n = X_flat.shape[0]
    if n % batch_size:
        raise ValueError("n_windows must be divisible by batch_size "
                         "to avoid recompiling make_bfgs on a shorter batch.")
    n_batches = n // batch_size
    eps_bfgs = jnp.asarray(bfgs_epsilon)
    eps_all = model.eps_all
    history, early_stopped = [], False

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated, no_decrease, prev_full = 0, 0, float('inf')

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            Xb = X_flat[idx]
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

            history.append(dict(epoch=epoch, batch=b, n=int(idx.shape[0]),
                                loss_before=l0, loss_after=l1,
                                loss_full=l_full, recon=rec, kl=kl))
            if verbose:
                print('epoch {} | batch {:>2d} | n={:<3d} | -ELBO '
                      '{:.4f} -> {:.4f} | full {:.4f} | recon {:.4f} '
                      '| KL {:.4f}'
                      .format(epoch, b, int(idx.shape[0]), l0, l1,
                              l_full, rec, kl))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d}\n'.format(epoch, b))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full -ELBO {:.4f} | updated {}/{} batches\n'
                  .format(epoch, l_full, n_updated, n_batches))
        if early_stopped:
            break

    return theta, history

if __name__ == '__main__':
    SEED, LATENT, HIDDEN, BETA = 42, 4, 5, 1.0
    L, D, N_WINDOWS = 6, 4, 512
    BATCH_SIZE, EPOCHS = 64, 8

    X = make_series(d=D, n_windows=N_WINDOWS, seed=7)
    X = (X - X.mean(axis=(0, 1))) / X.std(axis=(0, 1))
    print('data shape (n, L, d):', X.shape)

    model, X_flat = gen_vae(X, latent_dim=LATENT, hidden=HIDDEN, beta=BETA,
                            max_iter=40, seed=SEED)
    print('params       : {} (encoder {} + decoder {}) | Hessian {}x{} | latent {}'
          .format(model.n_params, model.n_enc,
                  model.n_params - model.n_enc,
                  model.n_params, model.n_params, LATENT))
    theta = model.theta0
    print('init full -ELBO: {:.4f}'
          .format(float(model.loss_full(theta, X_flat))))

    theta, hist = train_vae(model, theta, X_flat, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, bfgs_epsilon=1e-1,
                            seed=SEED, verbose=True, patience=0)
    print('final full -ELBO: {:.4f}'
          .format(float(model.loss_full(theta, X_flat))))
    th = jnp.asarray(theta)
    mu = model.encode_mu(th, X_flat)
    rec_flat = model.decode(th, mu)
    print('reconstruction MSE (z-score space): {:.4f}'
          .format(float(jnp.mean((X_flat - rec_flat) ** 2))))
