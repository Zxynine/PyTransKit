import numpy as np
from skimage.transform import radon, iradon
from joblib import Parallel, delayed #Used for RadonCDT, comes with skimage

import jax
import jax.numpy as jnp
import functools # For jax.jit static methods


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




def jit_static(*args, **kwargs):
    """Jit a function, preserve types, and make it a staticmethod."""
    return staticmethod(jit_func)
def jit_func(*args, **kwargs):
    """Wrap a function with jit and optional args."""
    def wrapper(func): return functools.wraps(func)(jax.jit(func, *args, **kwargs))
    return wrapper

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                    Core Transport Operations & Signal Representations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







class Signal(object):
    """Represents a Signal, its Domain, and its CDF"""
    def __iter__(self): return (self.x, self.s, self.C).__iter__()
    def __init__(self, x, sig):
        x = jnp.asarray(x, dtype=float)
        sig = jnp.asarray(sig, dtype=float)

        self.s = sig
        self.x = jnp.broadcast_to(x, self.s.shape)
        self.C, _ = _cdf(self.x, self.s)

        first_elem = jnp.expand_dims(self.s[..., 0], axis=-1)
        uniform = jnp.all(self.s == first_elem, axis=-1, keepdims=True)
        
        self.s = jnp.where(uniform, 1.0, self.s)
        self.C = jnp.where(uniform, self.x, self.C)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                          class wrappers to make transforms easier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




class TransformInterface(object):
    """ Unified base template that defines the general shape of a transform class """
    def __init__(self):
        self.fwd_map = self.inv_map = None
        self.is_mapped = False
    def find_mapping(self): raise NotImplementedError
    def apply_forward(self): raise NotImplementedError
    def apply_inverse(self): raise NotImplementedError
    def _check_mapped(self):
        if not self.is_mapped:
            raise AssertionError(f"Transform: '{type(self).__name__}' must be mapped first via .forward()")
        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class CDT(TransformInterface):
    def find_mapping(self, xref, yref, xsig, ysig):
        (self.fwd_map, self.fwd_mass), (self.inv_map, self.inv_mass) = CDT_Engine.Forward(xref, yref, xsig, ysig), CDT_Engine.Forward(xsig, ysig, xref, yref)
        self.is_mapped = True
        return self.fwd_map, self.inv_map

    def apply_forward(self, xnew, ynew):
        return CDT_Engine.Inverse((self.fwd_map, self.fwd_mass), xnew, ynew)

    def apply_inverse(self, xnew, ynew):
        return CDT_Engine.Inverse((self.inv_map, self.inv_mass), xnew, ynew)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






























#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               Work in progress transforms
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SCDT_new(TransformInterface):
    """Fully vectorized Signed Cumulative Distribution Transform."""
    def __init__(self, reference, x0=None):
        super().__init__()
        # Initialize domain grids and capture reference signal properties
        x0 = x0 if x0 is not None else jnp.linspace(0.0, 1.0, len(reference))
        self.ref = Signal(x0, reference)
        self.x = jnp.linspace(0.0, 1.0, self.ref.s.shape[-1])
        self.is_mapped = True 

    def transform(self, I):
        """Forward mapping: I_CDF_inverse(reference_CDF(x))"""
        return interp_batch(self.ref.C, _cdf(self.x, I)[0], self.x)

    def itransform(self, Ihat):
        """Inverse mapping coordinate projection."""
        return interp_batch(self.x, Ihat, self.ref.C)

    def stransform(self, I, x=None):
        """Splits an arbitrary signal into components and transforms them in parallel."""
        if x is not None: self.x = jnp.broadcast_to(jnp.asarray(x, dtype=float), I.shape)
        
        # Jordan Decomposition
        I_split = ((jnp.abs(I) + I) * 0.5, (jnp.abs(I) - I) * 0.5)
        
        # Parallelized mass extraction and forward transformation via map
        Iposhat, Ineghat = map(self.transform, I_split)
        m_pos, m_neg = map(lambda arr: jnp.sum(arr, axis=-1), I_split)

        # Guard against zero mass boundaries
        Iposhat = jnp.where(m_pos[..., None] > 1e-7, Iposhat, 0.0)
        Ineghat = jnp.where(m_neg[..., None] > 1e-7, Ineghat, 0.0)

        return Iposhat, Ineghat, m_pos, m_neg

    def istransform(self, Iposhat, Ineghat, Masspos, Massneg):
        """Synthesizes and reconstructs the signed signal scale"""
        m_pos, m_neg = map(lambda m: jnp.expand_dims(m, axis=-1), (Masspos, Massneg))
        Ipos, Ineg = map(self.itransform, (Iposhat, Ineghat))
        return jnp.where(m_pos > 0.0, Ipos*m_pos, 0.0) - jnp.where(m_neg > 0.0, Ineg*m_neg, 0.0)

    def calc_scdt(self, sig1, t1, s0, t0):
        """Unified runner returning a horizontally stacked mapping block."""
        Ipos, Ineg, mpos, mneg = SCDT_new(s0, t0).stransform(sig1, t1)
        return jnp.concatenate((Ipos, Ineg), axis=-1), mpos, mneg
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RadonCDT(TransformInterface):
    """ Parallelized Radon Cumulative Distribution Transform. """
    def __init__(self, theta=None, n_jobs=-1):
        super().__init__()
        self.theta = jnp.arange(180) if theta is None else jnp.asarray(theta)
        self.n_jobs = n_jobs
        self.cdt = CDT()

    def _project_and_batch(self, images, target_shape_reference=None):
        """Parallelizes CPU Radon projections and formats for JAX broadcasting."""
        img_np, theta_np = np.asarray(images), np.asarray(self.theta)

        if img_np.ndim > 2:
            flat_shape = img_np.shape[:-2]
            sinograms = Parallel(n_jobs=self.n_jobs)(
                delayed(radon)(img, theta=theta_np, circle=False) 
                for img in img_np.reshape(-1, img_np.shape[-2], img_np.shape[-1])
            )
            sinograms = np.transpose(np.array(sinograms), (0, 2, 1))
            return jnp.asarray(sinograms.reshape(flat_shape + sinograms.shape[1:]), dtype=float)
        
        if target_shape_reference is not None and len(np.unique(img_np)) == 1:
            ref = np.asarray(target_shape_reference)
            slice_2d = ref if ref.ndim == 2 else ref[0, 0] if ref.ndim == 4 else ref[0]
            return jnp.ones(radon(slice_2d, theta=theta_np, circle=False).shape[0], dtype=float)
        
        return jnp.asarray(radon(img_np, theta=theta_np, circle=False).T, dtype=float)

    def find_mapping(self, x0_range, sig0, x1_range, sig1):
        """Computes forward and inverse transform maps in parallel. Supports implicit batch dimentions."""
        # Extract projection sinograms
        print(sig1.shape)
        rad0_batch = self._project_and_batch(sig0, target_shape_reference=sig1)
        rad1_batch = self._project_and_batch(sig1)
        # Generate linear mapping coordinate vectors based on projection tracking dimensions
        x0 = jnp.linspace(x0_range[0], x0_range[1], rad0_batch.shape[-1]) # Unpack range bounds explicitly to keep jnp.linspace happy
        x1 = jnp.linspace(x1_range[0], x1_range[1], rad1_batch.shape[-1])

        # Route arrays directly into your vectorized JAX transport core
        self.fwd_map, self.inv_map = self.cdt.find_mapping(x0, rad0_batch, x1, rad1_batch)
        self.is_mapped = True
        
        # Swap axes to return data in your standard (..., Projections, Angles) space representation
        return jnp.swapaxes(self.fwd_map, -2, -1), jnp.swapaxes(self.inv_map, -2, -1)

    def _warp_and_reconstruct(self, input_data, xnew_range, cdt_method):
        """Unified internal inverse reconstruction pipeline supporting implicit batch dimensions."""
        self._check_mapped()
        is_single_img = (input_data.ndim == 2)
        
        batch_data = self._project_and_batch(input_data, target_shape_reference=input_data) if is_single_img else jnp.asarray(input_data, dtype=float)
        xnew_grid = jnp.linspace(xnew_range[0], xnew_range[1], batch_data.shape[-1])

        warped_batch = cdt_method(xnew_grid, batch_data)
        sinogram_np = np.asarray(jnp.swapaxes(warped_batch, -2, -1))
        
        # Parallelize the back-projection reconstruction phase across cores if handling a batch
        if sinogram_np.ndim > 2:
            flat_shape = sinogram_np.shape[:-2]
            recons = Parallel(n_jobs=self.n_jobs)(
                delayed(iradon)(s, theta=np.asarray(self.theta), circle=False, filter_name='hann') 
                for s in sinogram_np.reshape(-1, sinogram_np.shape[-2], sinogram_np.shape[-1]) # Uses flat sinos
            )
            return jnp.asarray(np.array(recons).reshape(flat_shape + recons.shape))
        return jnp.asarray(iradon(sinogram_np, theta=np.asarray(self.theta), circle=False, filter_name='hann'))

    def apply_forward(self, xnew_range, ynew_sig):
        return self._warp_and_reconstruct(ynew_sig, xnew_range, self.cdt.apply_forward)

    def apply_inverse(self, xnew_range, ynew_ref):
        return self._warp_and_reconstruct(ynew_ref, xnew_range, self.cdt.apply_inverse)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class PoissonCDT(TransformInterface):
    """
    Non-iterative, N-Dimensional Optimal Transport Mapping Engine.
    Solves the linearized Monge-Ampère equation instantly via FFT spectral division.
    Works natively on arrays of any matching N-D shape (e.g., [H, W], [D, H, W], etc.)
    """
    def __init__(self): super().__init__()

    def find_mapping(self, xref, yref, xsig, ysig):
        """
        Calculates the omnidirectional displacement fields directly. \n
        Returns the forward vector list (Ref -> Target) and inverse list (Target -> Ref).
        Args:
            xref: Domain coordinates of reference signal (e.g., 1D vector or N-D mesh)
            yref: Density values of reference signal
            xsig: Domain coordinates of target signal 
            ysig: Density values of target signal
        """
        self.xref = jnp.asarray(xref, dtype=float)
        self.xsig = jnp.asarray(xsig, dtype=float)
        
        # Ensure absolute mass conservation across both density tensors
        p = jnp.asarray(yref, dtype=float) / jnp.sum(yref)
        q = jnp.asarray(ysig, dtype=float) / jnp.sum(ysig)
        
        # Compute the differential source-minus-target charge mapping
        # Project the distribution variations into frequency space via N-D FFT
        charge_fft = jnp.fft.fftn(p - q)
        
        # Generate the analytical eigenvalue kernel for a discrete N-D Laplacian
        frequencies = [jnp.fft.fftfreq(s) for s in p.shape]
        mesh = jnp.meshgrid(*frequencies, indexing='ij')
        
        # Discrete finite-difference central Laplacian kernel
        laplacian_kernel = sum(4.0 * (jnp.sin(jnp.pi * m) ** 2) for m in mesh)
        laplacian_kernel = jnp.where(laplacian_kernel == 0.0, 1.0, laplacian_kernel)
        
        # Extract the scalar potential map Φ by inverting the spectral division
        phi = jnp.real(jnp.fft.ifftn(-charge_fft / laplacian_kernel))
        
        # Compute true physical coordinate sample steps (dx) for each axis (handles the scaling factor)
        spacings = []
        for axis in range(p.ndim):
            # Isolate the coordinate array for the current axis and find its uniform step size
            axis_coords = self.xsig if p.ndim == 1 else self.xsig[axis]
            # Take the step delta between index 1 and index 0
            spacings.append(axis_coords[1] - axis_coords[0])

        # Extract omnidirectional spatial gradients (The physical velocity fields)
        # Forward map pushes mass from Ref -> Target; Inverse map pulls Target -> Ref
        self.fwd_map = jnp.gradient(phi, *spacings)
        if p.ndim == 1: self.fwd_map = [self.fwd_map] # Protect formatting: if 1D, wrap gradient output in a single-element list
        self.inv_map = [-1.0 * axis_grad for axis_grad in self.fwd_map]
        
        self.is_mapped = True
        return self.fwd_map, self.inv_map

    def apply_forward(self, coordinates=None):
        """Warps domain coordinates forward along the physical velocity paths."""
        self._check_mapped()
        return [c + field for c, field in zip(coordinates or [self.xref], self.fwd_map)]

    def apply_inverse(self, coordinates=None):
        """Warps domain coordinates backward into the reference layout."""
        self._check_mapped()
        return [c + field for c, field in zip(coordinates or [self.xsig], self.inv_map)]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class GaussianCDT:
    """
    Non-iterative, Multi-Dimensional Closed-Form Transport Mapping Engine.
    Leverages Gaussian Monge mappings to account for cross-axial transport natively.
    Fully compatible with JAX vmap, jit, and autodiff.
    """
    def __init__(self):
        self.is_mapped = False

    @staticmethod
    def _matrix_sqrt(mat):
        """Computes the unique positive-definite matrix square root."""
        # Using Eigh because covariance matrices are symmetric positive-definite
        eigenvalues, eigenvectors = jnp.linalg.eigh(mat)
        # Clip negative eigenvalues due to numerical precision floors
        eigenvalues = jnp.maximum(eigenvalues, 1e-10)
        return eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T

    def find_mapping(self, x_grid, yref, ysig):
        """
        Computes the global cross-axial affine transport matrices.
        Args:
            x_grid: Physical coordinate mesh of shape [D, H, W] or [D, N]
                    where D is the number of spatial dimensions.
            yref: Density grid of reference signal (e.g., shape [H, W])
            ysig: Density grid of target signal (e.g., shape [H, W])
        """
        # 1. Flatten spatial grid dimensions to [D, total_pixels]
        spatial_dim = x_grid.shape[0]
        flat_coords = x_grid.reshape(spatial_dim, -1)
        
        p = yref.ravel() / jnp.sum(yref)
        q = ysig.ravel() / jnp.sum(ysig)

        # 2. Compute N-Dimensional Spatial Means (Centroids)
        self.mu_ref = jnp.sum(flat_coords * p, axis=-1)
        self.mu_sig = jnp.sum(flat_coords * q, axis=-1)

        # 3. Compute N-Dimensional Spatial Covariances (Cross-axial spreads)
        centered_ref = flat_coords - self.mu_ref[:, None]
        centered_sig = flat_coords - self.mu_sig[:, None]
        
        # Weighted outer products for true covariance tracking
        Sigma_ref = (centered_ref * p) @ centered_ref.T + jnp.eye(spatial_dim) * 1e-6
        Sigma_sig = (centered_sig * q) @ centered_sig.T + jnp.eye(spatial_dim) * 1e-6

        # 4. Compute Closed-Form Geometric Transport Matrix (A)
        Sigma_ref_sqrt = self._matrix_sqrt(Sigma_ref)
        Sigma_ref_inv_sqrt = jnp.linalg.inv(Sigma_ref_sqrt)
        
        inner_mat = Sigma_ref_sqrt @ Sigma_sig @ Sigma_ref_sqrt
        inner_sqrt = self._matrix_sqrt(inner_mat)
        
        # Final Forward mapping matrix
        self.A_fwd = Sigma_ref_inv_sqrt @ inner_sqrt @ Sigma_ref_inv_sqrt
        
        # Perfect Symmetrical Inverse Matrix (Zero iterations required)
        self.A_inv = jnp.linalg.inv(self.A_fwd)
        
        self.is_mapped = True
        return self.A_fwd, self.mu_ref, self.mu_sig

    def apply_forward(self, coordinates):
        """Maps any N-D coordinate array from Ref domain -> Target domain."""
        # coordinates shape expected: [D, ...]
        orig_shape = coordinates.shape
        flat_coords = coordinates.reshape(orig_shape[0], -1)
        
        # Affine Warp: T(x) = A*(x - mu_ref) + mu_sig
        centered = flat_coords - self.mu_ref[:, None]
        warped = self.A_fwd @ centered + self.mu_sig[:, None]
        
        return warped.reshape(orig_shape)

    def apply_inverse(self, coordinates):
        """Maps any N-D coordinate array from Target domain -> Ref domain."""
        orig_shape = coordinates.shape
        flat_coords = coordinates.reshape(orig_shape[0], -1)
        
        # Symmetrical Inverse Warp: T^-1(x) = A^-1*(x - mu_sig) + mu_ref
        centered = flat_coords - self.mu_sig[:, None]
        warped = self.A_inv @ centered + self.mu_ref[:, None]
        
        return warped.reshape(orig_shape)













































def _MatchSignals(x0,y0, x1,y1):
    pass




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
        psignal = jnp.where(signal > 0, signal, 1e-9)
        nsignal = jnp.where(signal < 0, -signal, 1e-9)
        return (psignal, nsignal)
    
    @staticmethod
    @jax.jit
    def Inverse(signal, x=None):
        x = x if x is not None else jnp.linspace(0,1, signal.shape[-1])
        inv = jnp.interp(jnp.linspace(0,1, signal.shape[-1]), signal, x)
        return inv/inv[-1]

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




class CDT_Engine(ITransportEngine):
    @staticmethod
    @jax.jit
    def Forward(xref, yref, xsig, ysig):
        xref, yref, xsig, ysig = CDT_Engine.Sanitize(xref, yref, xsig, ysig)
        CDF0 = _trapcumint(yref, xref) + 1e-7
        CDF1 = _trapcumint(ysig, xsig) + 1e-7
        MAP = interp_batch(CDF0/CDF0[..., -1:], CDF1/CDF1[..., -1:], jnp.linspace(0,1, xsig.shape[-1]))
        MASS = jnp.sum(ysig, axis=-1, keepdims=True)
        return (MAP, MASS) #this is the state
    
    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref, N=None):
        xref, yref = CDT_Engine.Sanitize(xref, yref)
        domain = (xref) if (N is None) else (jnp.linspace(0,1, N))
        MAP, MASS = state

        J = jnp.clip(jnp.gradient(MAP, xref, axis=-1), 1e-7)
        # J = _Jacobian(MAP, xref)
        warped = interp_batch(domain, MAP, yref/J)
        return _normalize(warped, MASS)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SCDT_Engine(ITransportEngine):
    @staticmethod
    @jax.jit
    def Forward(xref, yref, xsig, ysig):
        xref, yref, xsig, ysig = SCDT_Engine.Sanitize(xref, yref, xsig, ysig)
        pref, nref = SCDT_Engine.Split(yref)
        psig, nsig = SCDT_Engine.Split(ysig)
        PMAP, PMASS = CDT_Engine.Forward(xref, pref, xsig, psig)
        NMAP, NMASS = CDT_Engine.Forward(xref, nref, xsig, nsig)
        return ((PMAP, NMAP), (PMASS, NMASS))
    
    @staticmethod
    @jax.jit
    def Inverse(state, xref, yref):
        xref, yref = CDT_Engine.Sanitize(xref, yref)
        (PMAP, NMAP), (PMASS, NMASS) = state
        pref, nref = SCDT_Engine.Split(yref)
        PSIG = CDT_Engine.Inverse((PMAP, PMASS), xref, pref)
        NSIG = CDT_Engine.Inverse((NMAP, NMASS), xref, nref)
        SIG = PSIG - NSIG
        # Re-normalise to its absolute mass.
        return _normalize(SIG, PMASS+NMASS)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RadonCDT_Engine(ITransportEngine):
    epsilon = 1e-7
    n_jobs = -1 # Or set to your number of cores

    @staticmethod
    def _project(img_batch, theta=None):
        """Standardized CPU projection with Parallel processing."""
        theta = np.asarray(theta) if theta is not None else np.arange(180)

        img_arr = np.asarray(img_batch)
        img_arr = (
            img_arr[np.newaxis, ...] if (img_arr.ndim == 2) else
            img_arr.reshape(-1, img_arr.shape[-2], img_arr.shape[-1]) if (img_arr.ndim > 3)
            else img_arr
        )
        img_arr = img_arr/np.sum(img_arr)
        delayed_radon = delayed(radon)
        # Use joblib to parallelize the radon calculation
        projections = Parallel(n_jobs=RadonCDT_Engine.n_jobs)(
            delayed_radon(img, theta=theta, circle=False, preserve_range=True) 
            for img in img_arr
        )
        
        # Returns shape: (Batch, Angles, Projections)
        return jnp.asarray(projections).transpose(0, 2, 1) #

    @staticmethod
    def _backproject(sino_batch, output_size, theta=None):
        """Parallelized backprojection."""
        theta = np.asarray(theta) if theta is not None else np.arange(180)
        delayed_iradon = delayed(iradon)
        # Note: input is (Batch, Projections, Angles), we need to swap back for iradon
        recons = Parallel(n_jobs=RadonCDT_Engine.n_jobs)(
            delayed_iradon(sino.T, theta=theta, circle=False, filter_name='ramp', output_size=output_size) 
            for sino in sino_batch
        )
        return jnp.asarray(recons)
    
    @staticmethod
    def Forward(xref, yref, xsig, ysig, theta=None):
        # Prepare inputs on CPU first
        rad0 = RadonCDT_Engine._project(yref, theta=theta)
        rad1 = RadonCDT_Engine._project(ysig, theta=theta)

        # Now move to JAX for the accelerated CDT mapping
        num_proj_bins = rad0.shape[-1] if rad0.ndim == 3 else rad0.shape[0]
        x0 = jnp.linspace(xref[0], xref[1], num_proj_bins)
        x1 = jnp.linspace(xsig[0], xsig[1], num_proj_bins)
        
        return CDT_Engine.Forward(x0, rad0, x1, rad1)

    @staticmethod
    def Inverse(state, xref, yref, N=None, theta=None):
        rad0 = RadonCDT_Engine._project(yref, theta=theta)

        num_proj_bins = rad0.shape[-1] if rad0.ndim == 3 else rad0.shape[0]
        x0 = jnp.linspace(xref[0], xref[1], num_proj_bins)
        

        warped = CDT_Engine.Inverse(state, x0, rad0, N=N)
        reconstructed = RadonCDT_Engine._backproject(np.asarray(warped), yref.shape[-1], theta=theta)
        return jnp.asarray(reconstructed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
