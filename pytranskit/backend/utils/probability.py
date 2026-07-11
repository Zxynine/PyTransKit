import jax
import jax.numpy as jnp



# @jax.jit
# def _pad_diff(array, axis=-1, clip_floor=None):
#     """Computes finite differences and pads the trailing element safely to maintain shape."""
#     d_raw = jnp.diff(array, axis=axis)
#     d_padded = jnp.concatenate([d_raw, d_raw[..., -1:]], axis=axis)
#     return d_padded if clip_floor is None else jnp.clip(d_padded, clip_floor, None)





@jax.jit
def _vmapinterp(x, xp, fp): return jax.vmap(jnp.interp, in_axes=(0, 0, 0))(x,xp,fp)
@jax.jit
def _trapcumsum(sig, delta=1): return jnp.cumulative_sum((0.5*(sig[...,:-1]+sig[...,1:])*delta), axis=-1, include_initial=True)
@jax.jit
def _trapcumint(ysig, xsig): return jnp.cumulative_sum((0.5*(ysig[...,:-1]+ysig[...,1:])*jnp.diff(xsig, axis=-1)), axis=-1, include_initial=True)



@jax.jit
def _cdf(xsig, ysig):
    """Calculates a normalized cumulative distribution function via trapezoidal integration."""
    cumsum = _trapcumint(ysig, xsig)
    mass = cumsum[...,-1:]
    safecdf = jnp.where(mass == 0.0, cumsum, cumsum/mass)
    return safecdf, mass

@jax.jit
def _normalize(sig, scale=1):
    mass = jnp.sum(jnp.abs(sig), axis=-1, keepdims=True)
    return sig/mass * scale, mass


@jax.jit
def interp_batch(x, xp, fp):
    """Evaluates 1D interpolations globally across arbitrary batches for all inputs."""
    def Converter(a): return jnp.stack(a) if isinstance(a, (list,tuple)) else jnp.asarray(a, dtype=float)
    x, xp, fp = map(Converter, (x, xp, fp))
    batch_shape  = jnp.broadcast_shapes(x.shape[:-1], xp.shape[:-1], fp.shape[:-1])

    # Broadcast leading batch dimensions independently from tracking signal lengths
    x, xp, fp = map(lambda a: jnp.broadcast_to(a, batch_shape+(a.shape[-1],) ), (x, xp, fp))

    # Flatten all leading batch dimensions to support N-D inputs
    x, xp, fp = map(lambda a: a.reshape(-1, a.shape[-1]), (x, xp, fp))
    return _vmapinterp(x, xp, fp).reshape(batch_shape+(x.shape[-1],))



