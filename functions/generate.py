from typing import Optional
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

#@jax.jit
def get_generator(max_length: int = 300,
                         max_fourier_order: int = 3,
                         max_period_number: int = 5,
                         random_periods: bool = False,
                         random_changepoints: bool = False,
                         key: Optional[jnp.array] = None):
    # Seasonality
    x = jnp.arange(max_length)
    o = jnp.arange(1, max_fourier_order+1)
    if random_periods:
        with numpyro.plate("P", max_period_number):
            key, sub_key = jax.random.split(key)
            p = numpyro.sample("p", dist.Uniform(low=7, high=max_length//2), rng_key=sub_key).astype(int)
    else:
        p = jnp.array([7, 15, 31, 64, 122, 90, 150])[:max_period_number]
        
    arg = 2*jnp.pi*o[:, jnp.newaxis]/p
    arg = x[:, jnp.newaxis]*arg.reshape(arg.size)
    S_X = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)

    # Trend
    T = max_length
    if random_changepoints:
        key, sub_key = jax.random.split(key)
        S = numpyro.sample("S", dist.Uniform(low=1, high=10).astype(int), rng_key=sub_key)
    else:
        S = 50
    step = T/(S+1)
    s = jnp.arange(start=0, stop=T, step=step).astype(int)[1:]
    f = jax.vmap(lambda x: jnp.where(jnp.arange(T) >= x, jnp.ones(T), jnp.zeros(T)), 0, 0)
    A = f(s).T

    t = jnp.arange(T)
    k = t[:, jnp.newaxis]
    m = jnp.ones_like(k)

    A = A*(t[:, jnp.newaxis]-s)/T
    T_X = jnp.concatenate((A, k, m), axis=1)

    # Holidays
    h_key = jax.random.PRNGKey(392586)
    D = jax.random.uniform(h_key, (50,), minval=0, maxval=500).astype(int)
    H_X = jax.nn.one_hot(D, 500)[:, :max_length].T

    return key, jnp.concatenate((S_X, T_X, H_X), axis=1)

def generate_time_series(key, 
                         X, 
                         batch_size: int = 1):
    with numpyro.plate("B", batch_size):
        key, sub_key = jax.random.split(key)
        w = numpyro.sample("w", dist.Bernoulli(0.1*jnp.ones((1, X.shape[1]))).to_event(1), rng_key=sub_key)
        w_scale = numpyro.sample("w_scale", dist.Normal(jnp.zeros((1, X.shape[1]))).to_event(1), rng_key=key)
    return jnp.einsum("ij,bj->bi", X, w), w

