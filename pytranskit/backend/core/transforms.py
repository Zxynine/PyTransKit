import numpy as np
import jax
import jax.numpy as jnp
import functools # For jax.jit static methods
from typing import NamedTuple, Tuple, Any, Callable
import math


from ..utils.probability import _normalize, _cdf, interp_batch


"""
The design of the following code uses a Strategy pattern (Solvers). The code uses 
inheritance from ITransportEngine to standardise interactions with solvers. The solvers 
themselves are static classes (only containing static methods) as to enforce functional 
purity. Transformer classes will be created that use the functions of whichever solvers 
are passed to them. The transformers will be responsible for providing some of the 
benefits of OOP while delegating the actual implementation details to the transport engines.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                    Core Transport Operations & Signal Representations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TransportState(NamedTuple):
    """ Self-contained context envelope. Contains the map and any additional data needed for reconstruction. """
    transport_map: jnp.ndarray  # Core displacement mapping / coordinates
    target_mass: jnp.ndarray    # Signal mass metadata 


class ITransportEngine:
    """Abstract Base Class interface for all transport math engines."""
    def Forward(self, xref, yref, xsig, ysig) -> TransportState: raise NotImplementedError
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
        return TransportState(MAP, MASS)
    
    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref, N=None):
        xref, yref = CDT_Engine.Sanitize(xref, yref)
        domain = (xref) if (N is None) else (jnp.linspace(0,1, N))
        MAP, MASS = state

        J = jnp.clip(jnp.abs(jnp.gradient(MAP, xref, axis=-1)), 1e-7)
        warped = interp_batch(domain, MAP, yref/J)
        return _normalize(warped, MASS)[0]


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
        return _normalize(SIG, PMASS+NMASS)[0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RadonCDT_Engine(ITransportEngine):
    @staticmethod
    @jax.jit
    def Forward(xref, yref, xsig, ysig, theta=None):
        rad0 = jax_radon(yref, theta=theta) + 1e-15
        rad1 = jax_radon(ysig, theta=theta) + 1e-15
        
        x0 = jnp.linspace(xref[0], xref[1], rad0.shape[-1])
        x1 = jnp.linspace(xsig[0], xsig[1], rad0.shape[-1])
        
        return CDT_Engine.Forward(x0, rad0, x1, rad1)

    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref, theta=None):
        rad0 = jax_radon(yref, theta=theta) + 1e-15

        x0 = jnp.linspace(xref[0], xref[1], rad0.shape[-1])
        
        warped = CDT_Engine.Inverse(state, x0, rad0) # (Batch, Angles, Projections)
        return jax_iradon(warped - 1e-15, output_size=yref.shape[-1], theta=theta) 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~












#TODO: Move into the frontend section. This was mostly for Ivan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Transformers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This is essentially a factory pattern that wraps the transformer around whatever engine you want.
class Transformer:
    def __init__(self, engine:ITransportEngine, xref=None, yref=None):
        self.Engine = engine
        self.ref_bind(xref, yref)
    
    def ref_bind(self, xref, yref):
        self.xref = xref
        self.yref = yref

    def Forward(self, xsig, ysig):
        return self.Engine.Forward(self.xref, self.yref, xsig,ysig)
    
    def Inverse(self, state: TransportState, xnew=None):
        xref = xnew if xnew is not None else self.xref
        return self.Engine.Inverse(state, xref, self.yref)

    @staticmethod
    def Get_CDT(xref=None, yref=None): return Transformer(CDT_Engine, xref, yref)
    @staticmethod
    def Get_SCDT(xref=None, yref=None): return Transformer(SCDT_Engine, xref, yref)
    @staticmethod
    def Get_RCDT(xref=None, yref=None): return Transformer(RadonCDT_Engine, xref, yref)












    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                    Custom radon transform implementation for JAX.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@jax.jit
def jax_radon(img_batch, theta=None):
    """
    High-throughput Radon transformation

    """
    theta = jnp.arange(180) if theta is None else jnp.asarray(theta)
    img_arr = jnp.asarray(img_batch)
    # Normalize and force shape to (Batch, H, W)
    h, w = img_arr.shape[-2], img_arr.shape[-1]
    img_arr = img_arr.reshape(-1, h, w) if img_arr.ndim > 2 else img_arr[None, ...]
    
    # Geometry Setup - Single static coordinate plane
    target_bins = int(math.floor(math.sqrt(2) * max(h, w)) + 1)
    mid = (target_bins) / 2.0
    X, Y = jnp.meshgrid(*[jnp.linspace(-mid, mid, target_bins)] * 2)
    
    # Pre-calculate trig constants for the inner scan loop
    rad = jnp.deg2rad(theta)
    cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)

    # Shape layout: (Angles, 2, target_bins, target_bins)
    all_coords = jax.vmap(lambda c, s: jnp.stack([-s * X + c * Y + mid, c * X + s * Y + mid], axis=0))(cos_a, sin_a)


    # Padding dimensions
    ph, pw = (target_bins - h) // 2, (target_bins - w) // 2
    pad_width = ((ph, target_bins - h - ph), (pw, target_bins - w - pw))



    # Process one image across all angles sequentially
    def process_single_image(img):
        padded = jnp.pad(img, pad_width)
        
        # Core High-Throughput Kernel (Executes exactly ONE projection slice at a time)
        def single_projection_step(_, coords_at_angle):
            # 'padded' is read statically from parent scope without jnp.repeat overhead
            rotated_plane = jax.scipy.ndimage.map_coordinates(padded, coords_at_angle, order=1, cval=0.0)
            projection = jnp.sum(rotated_plane, axis=-1)
            return None, projection
        
        # Scan exclusively over the coordinates array
        _, sinogram = jax.lax.scan(single_projection_step, None, all_coords)
        return sinogram

    # Vectorize across the whole batch instantly 
    out_sinograms = jax.vmap(process_single_image)(img_arr)
    return out_sinograms

@functools.partial(jax.jit, static_argnames=['output_size'])
def jax_iradon(sino_batch, output_size=None, theta=None):
    """
    CPU-Backend Safe Filtered Back-Projection (FBP).
    Uses lax.scan instead of angle vmap to prevent YNNPACK library crashes.
    """
    theta = jnp.arange(180) if theta is None else jnp.asarray(theta)
    sino_arr = jnp.asarray(sino_batch)
    
    # Standardize to 3D: (Batch, Angles, Projections)
    sino_arr = sino_arr[None, ...] if sino_arr.ndim == 2 else sino_arr
    if sino_arr.shape[-2] != len(theta) and sino_arr.shape[-1] == len(theta):
        sino_arr = jnp.transpose(sino_arr, (0, 2, 1))
        
    target_bins = sino_arr.shape[-1]
    mid = (target_bins - 1) / 2.0
    
    # Standard Fourier Ramp Filter setup
    frequencies = jnp.fft.fftfreq(target_bins)
    ramp_filter = jnp.abs(frequencies)
    
    # Pre-calculate trig constants and package them together for the loop scan
    rad = jnp.deg2rad(theta)
    cos_sin_pairs = jnp.stack([jnp.cos(rad), jnp.sin(rad)], axis=1) # (Angles, 2)
    
    # Geometry coordinate mesh grid
    X, Y = jnp.meshgrid(*[jnp.linspace(-mid, mid, target_bins)] * 2)

    # Filter along the projection axis via 1D FFT
    filtered_sino = jnp.real(jnp.fft.ifft(jnp.fft.fft(sino_arr, axis=-1) * ramp_filter, axis=-1))

    # Helper function to reconstruct ONE sinogram
    def reconstruct_single_sinogram(single_sino):
        # Initial accumulation state for the backprojected image grid
        init_grid = jnp.zeros((target_bins, target_bins))
        
        # Package the angles data and its corresponding slice of the sinogram together
        loop_inputs = (single_sino, cos_sin_pairs)

        def scan_step(grid_carry, inputs):
            angle_slice, trig = inputs
            c, s = trig[0], trig[1]
            
            # Map coordinates for the current projection angle slice
            u_idx = (c * X + s * Y + mid)[None, ...]
            backprojection = jax.scipy.ndimage.map_coordinates(angle_slice, u_idx, order=1, cval=0.0)
            
            # Accumulate backprojections directly into the carry state grid
            return grid_carry + backprojection, None

        # Sequentially loop over angles. This avoids the CPU multithreaded YNNPACK parallel vector allocation bug.
        final_grid, _ = jax.lax.scan(scan_step, init_grid, loop_inputs)
        return final_grid

    # Standard clean vmap across the outer Batch dimension
    recons = jax.vmap(reconstruct_single_sinogram)(filtered_sino)
    
    # Normalize backprojections by the total count of angles and correct orientation
    recons = recons / len(theta)
    recons = recons * (jnp.pi / (2.0))
    recons = jnp.flip(recons, axis=1)
    
    # Programmatic crop back to original image grid size
    if output_size is not None:
        p = (target_bins - output_size) // 2
        recons = recons[:, p:p+output_size, p:p+output_size]
        
    return recons






