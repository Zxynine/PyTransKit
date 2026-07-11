import numpy as np
from scipy.fftpack import dct, idct

from .cdt import CDT
from ..utils import check_array, assert_equal_shape, signal_to_pdf, interp2d, griddata2d

from pytranskit.legacy.optrans.continuous.transforms import CDT_Engine

















import numpy as np
import jax
import jax.numpy as jnp
from scipy.fft import dct, idct

from .cdt import CDT
from ..utils import check_array, assert_equal_shape, signal_to_pdf, interp2d, griddata2d


class CLOT():
    """
    Continuous Linear Optimal Transport Transform.

    This uses Nesterov's accelerated gradient descent to remove the curl in the
    initial mapping, utilizing the accelerated JAX CDT_Engine for initialization.

    Parameters
    ----------
    lr : float (default=0.01)
        Learning rate.
    momentum : float (default=0.)
        Nesterov accelerated gradient descent momentum.
    decay : float (default=0.)
        Learning rate decay over each update.
    max_iter : int (default=300)
        Maximum number of iterations.
    tol : float (default=0.001)
        Stop iterating when change in cost function is below this threshold.
    verbose : int (default=0)
        Verbosity during optimization. 0=no output, 1=print cost,
        2=print all metrics.
    """
    def __init__(self, lr=0.01, momentum=0., decay=0., max_iter=300, tol=0.001,
                 verbose=0):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.is_fitted = False
        return

    def forward(self, sig0, sig1):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        lot : array, shape (2, height, width)
            LOT transform of input image sig1. First index denotes direction:
            lot[0] is y-LOT, and lot[1] is x-LOT.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Create regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))

        # Compute initial mapping using the new JAX CDT Engine
        f = self._get_initial_map(sig0, sig1)
        self.transport_map_initial_ = np.copy(f)
        self.displacements_initial_ = f - np.stack((yv, xv))

        # Initialise evaluation measures
        self.cost_ = []
        self.curl_ = []

        # Initialise derivative of cost function wrt f
        ft = np.zeros_like(f)

        # Initialise previous update (for Nesterov momentum)
        update_prev = np.zeros_like(f)
        f_prev = np.copy(f)

        for i in range(self.max_iter):
            # Save previous version of f before update
            f_prev = np.copy(f)

            # Nesterov momentum "look ahead"
            f -= self.momentum * update_prev

            # Jacobian and its determinant
            f0y, f0x = np.gradient(f[0])
            f1y, f1x = np.gradient(f[1])
            
            # Update evaluation measures
            cost = np.sum(((yv - f[0])**2 + (xv - f[1])**2) * sig0)
            self.cost_.append(cost)
            curl = 0.5 * (f0x - f1y)
            self.curl_.append(0.5 * np.sum(curl**2))

            # Print metrics
            if self.verbose:
                print(f'Iteration {i:>4} -- cost = {self.cost_[-1]:.4e}')
            if self.verbose > 1:
                print(f'... curl = {self.curl_[-1]:.4e}')

            # Divergence
            vx = np.gradient(-f[0] + yv, axis=1)
            uy = np.gradient(f[1] - xv, axis=0)
            div = vx + uy

            # Poisson solver
            div_dct = dct(dct(div, axis=0, norm='ortho'), axis=1, norm='ortho')
            denom = (2 * np.cos(np.pi * xv / w) - 2) + (2 * np.cos(np.pi * yv / h) - 2)
            denom[0, 0] = 1.  # Avoid division by zero
            div_dct /= denom
            lneg = -idct(idct(div_dct, axis=1, norm='ortho'), axis=0, norm='ortho')
            lnegy, lnegx = np.gradient(lneg)

            # Derivative of cost function wrt f
            ft[0] = (-f0x * lnegy + f0y * lnegx) / sig0
            ft[1] = (-f1x * lnegy + f1y * lnegx) / sig0

            # Update transport map (using local scheduling variable to avoid mutating self.lr)
            current_lr = self.lr * (1. / (1. + self.decay * i))
            update = self.momentum * update_prev + current_lr * ft
            update_prev = np.copy(update)
            f -= update

            # Early stopping check
            if i > 7 and (self.cost_[i-7] - self.cost_[i]) / self.cost_[0] < self.tol:
                break

        if self.verbose:
            print('FINAL METRICS:')
            print(f'-- cost = {self.cost_[-1]:.4e}')
            print(f'-- curl = {self.curl_[-1]:.4e}')

        self.transport_map_ = f_prev
        self.displacements_ = f_prev - np.stack((yv, xv))
        lot = self.displacements_ * np.sqrt(sig0)

        self.is_fitted = True
        return lot
    
    def _get_initial_map(self, sig0, sig1):
        """Get initial transport map utilizing vectorized CDT execution with pixel scaling."""
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        fill_val = min(sig0.min(), sig1.min())

        # 1. Horizontal Pass: Calculate clean PDFs over a [0, 1] normalized grid
        sum0 = jnp.clip(jnp.asarray(signal_to_pdf(sig0.sum(axis=0), epsilon=1e-4)), 1e-7, None)
        sum1 = jnp.clip(jnp.asarray(signal_to_pdf(sig1.sum(axis=0), epsilon=1e-4)), 1e-7, None)

        # EXTERNAL FIX: Define the input domains on a [0, 1] normalized scale
        x0_grid = jnp.linspace(0, 1, w)
        x1_grid = jnp.linspace(0, 1, w)
        
        # This will now cleanly execute through the unmodified CDT Engine
        map_x, _ = CDT_Engine.Forward(x0_grid, sum0, x1_grid, sum1)
        
        # Scale the resulting normalized map back to the native pixel domain [0, w-1]
        a = np.tile(np.array(map_x) * (w - 1), (h, 1))
        aprime = np.gradient(a, axis=1)

        # Compute a'(x)sig1(a(x),y) for all y
        siga_raw = aprime * interp2d(sig1, np.stack((yv, a)), fill_value=fill_val)

        # Normalize intermediate columns to safe vertical PDFs
        siga_clean = np.zeros_like(siga_raw)
        for col in range(w):
            siga_clean[:, col] = signal_to_pdf(siga_raw[:, col], epsilon=1e-4)

        # 2. Vertical Pass: Vectorized Column-Wise Processing
        sig0_t = jnp.asarray(sig0.T)
        siga_t = jnp.asarray(siga_clean.T)

        # EXTERNAL FIX: Set up the vertical tracking dimension on a [0, 1] grid 
        # and broadcast it to match the transposed batch size (w columns)
        col_grid_base = jnp.linspace(0, 1, h)
        col_grid = jnp.broadcast_to(col_grid_base, (w, h))
        
        # Batch transform across all columns simultaneously
        map_y_batched, _ = CDT_Engine.Forward(col_grid, sig0_t, col_grid, siga_t)
        
        # Retranspose and scale the normalized map back to vertical pixel domain [0, h-1]
        b = np.array(map_y_batched).T * (h - 1)

        # Clean up unmapped/empty boundary columns
        zero_column = np.array(np.where(np.mean(b, axis=0) == 0.))
        if len(zero_column) > 0 and zero_column.size > 0:
            for z in zero_column[0]:
                b[:, z] = b[:, z + 1] if z == 0 else b[:, z - 1]

        return np.stack((b, a))
    


    def apply_forward_map(self, transport_map, sig1):
        """
        Apply forward transport map.

        Parameters
        ----------
        transport_map : array, shape (2, height, width)
            Forward transport map.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        sig0_recon : array, shape (height, width)
            Reconstructed reference signal sig0.
        """
        transport_map = check_array(transport_map, ndim=3, dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32], force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig1, ['transport_map', 'sig1'])

        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        return detJ * interp2d(sig1, transport_map, fill_value=sig1.min())

    def apply_inverse_map(self, transport_map, sig0):
        """
        Apply inverse transport map.

        Parameters
        ----------
        transport_map : array, shape (2, height, width)
            Forward transport map. Inverse is computed in this function.
        sig0 : array, shape (height, width)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        transport_map = check_array(transport_map, ndim=3, dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32], force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig0, ['transport_map', 'sig0'])

        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        return griddata2d(sig0 / detJ, transport_map, fill_value=sig0.min())
