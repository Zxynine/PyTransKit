import numpy as np
from skimage.transform import radon, iradon
from joblib import Parallel, delayed #Used for RadonCDT, comes with skimage

import jax
import jax.numpy as jnp
import functools # For jax.jit static methods
from typing import NamedTuple, Tuple, Any, Callable
import math

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                          jax configuration for gpu/cpu targeting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_pipeline_hardware(target_hardware='gpu'):
    """  Sets global backend execution target ('gpu' or 'cpu')   \n
        Try to avoid setting this multiple times"""
    try:
        if (target_hardware is None): return
        jax.config.update('jax_platform_name', target_hardware.lower())
        jnp.linspace(0,1,10) #Force jax to make a computation to check it
        print(f"Pipeline target successfully configured for: {target_hardware.upper()}")
    except RuntimeError as e:
        print(f"Failed setting hardware backend choice: {e}")


# config.update("jax_enable_x64", True)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                          jax wrapped functions for vectorising code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This section is primarily for keeping code short and sweet elsewhere. Handles common args.

def IsSingle(v): return not isinstance(v, (list, tuple)) or hasattr(v, 'ndim') or isinstance(v, (str, dict))

def flatten_batch(arr, bshape): return jnp.broadcast_to(arr, bshape+(arr.shape[-1],)).reshape(-1, arr.shape[-1])
def inflate_batch(flat, bshape): return flat.reshape(bshape + (flat.shape[-1],))

@jax.jit
def _vmapinterp(x, xp, fp): return jax.vmap(jnp.interp, in_axes=(0, 0, 0))(x,xp,fp)
@jax.jit
def _trapcumsum(sig, delta=1): return jnp.cumulative_sum((0.5*(sig[...,:-1]+sig[...,1:])*delta), axis=-1, include_initial=True)
@jax.jit
def _trapcumint(ysig, xsig): return jnp.cumulative_sum((0.5*(ysig[...,:-1]+ysig[...,1:])*jnp.diff(xsig, axis=-1)), axis=-1, include_initial=True)

@jax.jit
def _pad_diff(array, axis=-1, clip_floor=None):
    """Computes finite differences and pads the trailing element safely to maintain shape."""
    d_raw = jnp.diff(array, axis=axis)
    d_padded = jnp.concatenate([d_raw, d_raw[..., -1:]], axis=axis)
    return d_padded if clip_floor is None else jnp.clip(d_padded, clip_floor, None)

@jax.jit
def interp_batch(x, xp, fp):
    """Evaluates 1D interpolations globally across arbitrary batches."""
    def Converter(a): return jnp.stack(a) if isinstance(a, (list,tuple)) else jnp.asarray(a, dtype=float)
    print(x.shape, xp.shape, fp.shape)
    x, xp, fp = map(Converter, (x, xp, fp))
    batch_shape  = jnp.broadcast_shapes(x.shape[:-1], xp.shape[:-1], fp.shape[:-1])

    # Broadcast leading batch dimensions independently from tracking signal lengths
    x, xp, fp = map(lambda a: jnp.broadcast_to(a, batch_shape+(a.shape[-1],) ), (x, xp, fp))

    # Flatten all leading batch dimensions to support N-D inputs
    x, xp, fp = map(lambda a: a.reshape(-1, a.shape[-1]), (x, xp, fp))
    return _vmapinterp(x, xp, fp).reshape(batch_shape+(x.shape[-1],))


@jax.jit
def _Jacobian(w, t): # Manual Jacobian (dw/dt)
    dwdt = (jnp.diff(w, axis=-1) + 1e-9) / (jnp.diff(t, axis=-1) + 1e-9)
    # Use padding to keep shape consistent
    return jnp.concatenate([dwdt, dwdt[..., -1:]], axis=-1)


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
    return sig/mass * scale


# @jax.jit
# def _pdf(sig, epsilon=1e-7):
#     sig = jnp.abs(sig) + epsilon




def jit_func(*args, **kwargs):
    """Wrap a function with jit and optional args."""
    def wrapper(func): return functools.wraps(func)(jax.jit(func, *args, **kwargs))
    return wrapper
def jit_static(*args, **kwargs):
    """Jit a function, preserve types, and make it a staticmethod."""
    return staticmethod(jit_func)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 Meta utility functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pack_nd_batches(data_dims: int, *args: jnp.ndarray):
    """Broadcasts and flattens leading dimensions. Returns flat args and metadata."""
    batch_shapes = [arr.shape[:-data_dims] if data_dims > 0 else arr.shape for arr in args]
    common_batch_shape = jnp.broadcast_shapes(*batch_shapes)

    # Aligns and flattens the arrays into the correct batch shape 
    BC_RESHAPE = lambda arg, trailing: jnp.broadcast_to(arg, common_batch_shape+trailing).reshape(-1, *trailing)

    flat_args = [
        BC_RESHAPE(arr, arr.shape[-data_dims:] if data_dims > 0 else ()) 
        for arr in args
    ]
        
    # Return the flat arrays, and the recipe to restore them later
    return flat_args, common_batch_shape

def unpack_nd_batches(flat_output: jnp.ndarray, original_batch_shape: tuple):
    """Instantly restores the original multidimensional batch structure."""
    # original_batch_shape handles the front, .shape[1:] handles the back
    return flat_output.reshape(original_batch_shape + flat_output.shape[1:])

# https://github.com/ott-jax/ott/blob/main/src/ott/utils.py

def chunked_vmap(fun: Callable, batch_size: int, out_axis: int = 0) -> Callable:
    """Wraps a function to process a flattened array sequentially in chunks along axis 0."""
    def wrapped(*args: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # Assumes pack_nd_batches was run first; all inputs share a unified total size on axis 0
        total_size = args[0].shape[0]
        num_chunks = total_size // batch_size
        remainder_size = total_size % batch_size
        
        # Core vectorization layer
        vmapped_fun = jax.vmap(fun, in_axes=0, out_axes=out_axis)
        
        # Process even chunks sequentially via lax.scan to cap memory consumption
        chunked_results = None
        if num_chunks > 0:
            even_elements = num_chunks * batch_size
            
            # Reshape from (Total, Spatial...) -> (Num_Chunks, Batch_Size, Spatial...)
            chunk_args = [
                arg[:even_elements].reshape(num_chunks, batch_size, *arg.shape[1:]) 
                for arg in args
            ]
            
            def body_fn(_, x_slice): return None, vmapped_fun(*x_slice, **kwargs)
            
            # Scan runs chunk-by-chunk under the hood, preventing memory spikes
            _, chunked_outputs = jax.lax.scan(body_fn, None, chunk_args)
            
            # Collapse the Scanned Chunk dimension and the Vmap Batch dimension back together
            flat_chunk_axis = jnp.moveaxis(chunked_outputs, 0, out_axis)
            chunked_results = jax.lax.collapse(flat_chunk_axis, out_axis, out_axis + 2)

        # Process remaining un-even elements in a single fallback vectorization step
        remainder_results = None
        if remainder_size > 0:
            remainder_args = [arg[num_chunks * batch_size:] for arg in args]
            remainder_results = vmapped_fun(*remainder_args, **kwargs)
            
        # Stitch chunks and remainders seamlessly back together
        if chunked_results is not None and remainder_results is not None:
            return jnp.concatenate([chunked_results, remainder_results], axis=out_axis)
        return chunked_results if chunked_results is not None else remainder_results

    return wrapped


def massive_nd_vmap(fun: Callable, batch_size: int, data_dims: int) -> Callable:
    """Combines N-D broadcasting and memory-safe chunking for any dataset size.
    
    Args:
        fun: Core function processing a single data element (Vector, Image, Volume, etc.).
        batch_size: Number of elements processed concurrently per loop step.
        data_dims: Total trailing axes belonging to the core data structural layout 
                   (e.g., 1 for Vectors, 3 for Images [H,W,C], 4 for Volumes [D,H,W,C]).
    """
    # Build the sequential execution pipeline
    chunked_core = chunked_vmap(fun, batch_size=batch_size, out_axis=0)
    
    @functools.wraps(fun)
    def wrapper(*args: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # Broadcaster / Flattener 
        flat_args, original_batch_shape = pack_nd_batches(data_dims, *args)
        
        # Memory-Capped Chunking
        flat_outputs = chunked_core(*flat_args, **kwargs)
        
        # Unpack and restore the outer batch shape
        final_shape = original_batch_shape + flat_outputs.shape[1:]
        return flat_outputs.reshape(final_shape)
        
    return wrapper

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                    Core Transport Operations & Signal Representations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



@jax.jit
def jax_radon(img_batch, theta=None):
    """
    Pure JAX forward Radon Transform.
    
    Parameters:
        image: 2D JAX array of shape (H, W)
        theta: 1D JAX array of angles in degrees
    Returns:
        sinogram: 2D JAX array of shape (target_bins, len(theta))
    """
    theta = jnp.arange(180) if theta is None else jnp.asarray(theta)
    img_arr = jnp.asarray(img_batch)
    
    # Standardize shape to (Batch, H, W) and normalize
    h, w = img_arr.shape[-2], img_arr.shape[-1]
    img_arr = img_arr.reshape(-1, h, w) if img_arr.ndim > 2 else img_arr[None, ...]
    img_arr /= jnp.sum(img_arr, axis=(-2, -1), keepdims=True)
    
    # Geometry Setup
    target_bins = int(math.ceil(math.sqrt(2) * max(h, w)) + 1)
    mid = (target_bins - 1) / 2.0
    X, Y = jnp.meshgrid(*[jnp.linspace(-mid, mid, target_bins)] * 2)
    rad = jnp.deg2rad(theta)
    cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
    
    # Vectorized core: loops over image batches and angles natively
    def single_projection(img, c, s):
        ph, pw = (target_bins - h) // 2, (target_bins - w) // 2
        padded = jnp.pad(img, ((ph, target_bins - h - ph), (pw, target_bins - w - pw)))
        coords = jnp.stack([-s * X + c * Y + mid, c * X + s * Y + mid], axis=0)
        return jnp.sum(jax.scipy.ndimage.map_coordinates(padded, coords, order=1, cval=0.0), axis=0)


    # vmap(..., (None, 0, 0)) handles angles; outer vmap(..., (0, None, None)) handles batch
    return jax.vmap(jax.vmap(single_projection, in_axes=(None, 0, 0)), in_axes=(0, None, None))(img_arr, cos_a, sin_a)


@functools.partial(jax.jit, static_argnames=['output_size'])
def jax_iradon(sino_batch, output_size=None, theta=None):
    """
    Pure JAX Inverse Radon Transform.
    
    Parameters:
        sinogram: 2D JAX array of shape (target_bins, len(theta))
        theta: 1D JAX array of angles in degrees
        output_shape: Optional tuple of static ints (H, W) to crop the output
    Returns:
        reconstructed: 2D JAX array
    """
    theta = jnp.arange(180) if theta is None else jnp.asarray(theta)
    sino_arr = jnp.asarray(sino_batch)
    sino_arr = sino_arr[None, ...] if sino_arr.ndim == 2 else sino_arr
    
    # Standardize to (Batch, Angles, Projections)
    if sino_arr.shape[-2] != len(theta) and sino_arr.shape[-1] == len(theta):
        sino_arr = jnp.transpose(sino_arr, (0, 2, 1))
    target_bins = sino_arr.shape[-1]
    
    # Filter Setup & Geometry Definitions
    ramp_filter = jnp.abs(jnp.fft.fftshift(jnp.linspace(-1.0, 1.0, target_bins)))
    mid = (target_bins - 1) / 2.0
    X, Y = jnp.meshgrid(*[jnp.linspace(-mid, mid, target_bins)] * 2)
    rad = jnp.deg2rad(theta)
    cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
    
    # Core FBP step per angle
    def backproject_angle(angle_sino, c, s):
        u_idx = (c * X + s * Y + mid)[None, ...]
        return jax.scipy.ndimage.map_coordinates(angle_sino, u_idx, order=1, cval=0.0)

    # Process sinograms via FFT
    filtered_sino = jnp.real(jnp.fft.ifft(jnp.fft.fft(sino_arr, axis=-1) * ramp_filter, axis=-1))
    
    # Dual-vmap over batch (axis 0) and angles (axis 1)
    reconstructions = jax.vmap(jax.vmap(backproject_angle, in_axes=(0, 0, 0)), in_axes=(0, None, None))(filtered_sino, cos_a, sin_a)
    recons = jnp.flip(jnp.mean(reconstructions, axis=1) * (jnp.pi / (2.0 * len(theta))), axis=1)
    
    # Clean programmatic crop
    if output_size is not None:
        p = (target_bins - output_size) // 2
        recons = recons[:, p:p+output_size, p:p+output_size]
    return recons








@jax.jit
def batch_broadcast_inverse(f_batch, dom_f, dom_gf_batch):
    """
    f_batch: shape (Batch, N)
    dom_f: shape (N,)
    dom_gf_batch: shape (Batch, M)
    """
    # Align axes for 3D broadcasting: (Batch, M_ref, N_sig)
    f_3d = f_batch[:, None, :]          # Shape: (Batch, 1, N)
    dom_gf_3d = dom_gf_batch[:, :, None] # Shape: (Batch, M, 1)
    dom_f_3d = dom_f[None, None, :]      # Shape: (1, 1, N)
    
    # Create the boolean threshold mask
    mask = f_3d > dom_gf_3d
    
    # Fill invalid zones with infinity, valid zones with domain coordinates
    masked_dom = jnp.where(mask, dom_f_3d, jnp.inf)
    
    # Pull the minimum domain point across the signal domain axis (-1)
    gf = jnp.min(masked_dom, axis=-1)
    
    # Clean up the upper boundary edge case
    return jnp.where(jnp.isinf(gf), dom_f[-1], gf)










class TransportState(NamedTuple):
    """ Self-contained context envelope. Contains the map and any additional data needed for reconstruction. """
    transport_map: jnp.ndarray  # Core displacement mapping / coordinates
    target_mass: jnp.ndarray    # Signal mass metadata 







"""
The design of the following code uses a Strategy pattern (Solvers). The code uses 
inheritance from ITransportEngine to standardise interactions with solvers. The solvers 
themselves are static classes (only containing static methods) as to enforce functional 
purity. Transformer classes will be created that use the functions of whichever solvers 
are passed to them. The transformers will be responsible for providing some of the 
benefits of OOP while delegating the actual implementation details to the transport engines.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ITransportEngine:
    """Base interface for all transport math engines."""
    def Forward(self, xref, yref, xsig, ysig): raise NotImplementedError
    def Inverse(self, state, xref, yref): raise NotImplementedError

    @staticmethod
    @jax.jit
    def Sanitize(*args): return tuple((
            jnp.stack(arg) 
            if isinstance(arg, (list,tuple)) else 
            jnp.asarray(arg, dtype=float)
        ) for arg in args)
    
    @staticmethod
    @jax.jit
    def Split(signal):
        psignal = jnp.where(signal > 0, signal, 0)
        nsignal = jnp.where(signal < 0, -signal, 0)
        return (psignal, nsignal)
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




class CDT_Engine(ITransportEngine):
    @staticmethod
    @jax.jit
    def Forward(xref, yref, xsig, ysig):
        xref, yref, xsig, ysig = CDT_Engine.Sanitize(xref, yref, xsig, ysig)
        CDF0,_ = _cdf(xref, yref)
        CDF1,_ = _cdf(xsig, ysig)
        MAP = interp_batch(CDF0, CDF1, xsig)
        MASS = jnp.sum(ysig, axis=-1, keepdims=True)
        return TransportState(MAP, MASS) #this is the state
    
    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref, N=None):
        xref, yref = CDT_Engine.Sanitize(xref, yref)
        domain = (xref) if (N is None) else (jnp.linspace(0,1, N))
        MAP, MASS = state

        J = jnp.clip(jnp.abs(jnp.gradient(MAP, xref, axis=-1)), 1e-7)
        warped = interp_batch(domain, MAP, yref/J)
        return _normalize(warped, MASS)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SCDT_Engine(ITransportEngine):
    @staticmethod
    @jax.jit
    def Forward(xref, yref, xsig, ysig):
        xref, yref, xsig, ysig = SCDT_Engine.Sanitize(xref, yref, xsig, ysig)
        # pref, nref = SCDT_Engine.Split(yref)
        pref = nref = jnp.abs(yref)
        psig, nsig = SCDT_Engine.Split(ysig)
        PMAP, PMASS = CDT_Engine.Forward(xref, pref, xsig, psig)
        NMAP, NMASS = CDT_Engine.Forward(xref, nref, xsig, nsig)
        return TransportState((PMAP, NMAP), (PMASS, NMASS))
    
    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref):
        xref, yref = CDT_Engine.Sanitize(xref, yref)
        (PMAP, NMAP), (PMASS, NMASS) = state
        # pref, nref = SCDT_Engine.Split(yref)
        pref = nref = jnp.abs(yref)
        PSIG = CDT_Engine.Inverse((PMAP, PMASS), xref, pref)
        NSIG = CDT_Engine.Inverse((NMAP, NMASS), xref, nref)
        SIG = PSIG - NSIG
        # Re-normalise to its absolute mass.
        return _normalize(SIG, PMASS+NMASS)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RadonCDT_Engine(ITransportEngine):
    @staticmethod
    def Forward(xref, yref, xsig, ysig, theta=None):
        # 1. FIX: Pass the actual signal image batch (ysig) instead of the 1D domain coordinates (xsig)
        rad1 = jax_radon(ysig, theta=theta)
        rad0 = jax_radon(yref, theta=theta)
        
        x0 = jnp.linspace(xref[0], xref[1], rad0.shape[-1])
        x1 = jnp.linspace(xsig[0], xsig[1], rad0.shape[-1])
        
        # 3. Pass directly to the accelerated CDT solver layout
        return CDT_Engine.Forward(x0, rad0, x1, rad1)

    @staticmethod
    def Inverse(state, xref, yref, theta=None):
        rad0 = jax_radon(yref, theta=theta)

        x0 = jnp.linspace(xref[0], xref[1], rad0.shape[-1])
        
        # Warp the distribution maps
        warped = CDT_Engine.Inverse(state, x0, rad0) # (Batch, Angles, Projections)
        return jax_iradon(warped, output_size=yref.shape[-1], theta=theta) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
