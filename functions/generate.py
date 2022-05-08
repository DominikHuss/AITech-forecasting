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
                         key: Optional[jnp.array] = None):
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
    X = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)
    return X

def generate_time_series(key, 
                         X, 
                         batch_size: int = 1):
    with numpyro.plate("B", batch_size):
        key, sub_key = jax.random.split(key)
        w = numpyro.sample("w", dist.Bernoulli(0.1*jnp.ones((1, X.shape[1]))).to_event(1), rng_key=sub_key)
    return jnp.einsum("ij,bj->bi", X, w), w


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    key = jax.random.PRNGKey(2137)
    A = get_generator()
    Y, w = generate_time_series(key, A)
    print(Y.shape)
    plt.plot(list(range(300)), list(Y[0]))
    plt.savefig("test.png")

