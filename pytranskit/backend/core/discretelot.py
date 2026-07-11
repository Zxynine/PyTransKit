import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix





def _prep_inputs(X, a, Normalise=True):
    X = np.asarray(X)
    a = np.asarray(a)
    mass = np.sum(a)
    return (
        X if (X.ndim > 1) else X[:, None],
        a if (mass == 0 or not Normalise)  else a / mass
    )

def _compute_cost(X0,X1):
    return ((X0[:, None, :] - X1[None, :, :]) ** 2).sum(axis=-1)





class DiscreteLOT:
    """
    Discrete Linear Optimal Transport (LOT) transform.

    This class computes the LOT embedding of a target measure (X1, a1)
    with respect to a reference measure (X0, a0).
    
    Authors
    -------
    Mohammad Shifat-E-Rabbi 
    Adapted for pytranskit and research workflow.

    Original framework inspired by:
    Kolouri, Soheil et al. (Optimal Transport methods)

    Date
    ----
    Created: June 2026

    Parameters
    ----------
    normalize : bool, optional (default=True)
        Whether to normalize input weights a0 and a1.
    solver : str, optional (default="highs")
        Linear programming solver to use.
    """

    def __init__(self, normalize=True, solver="highs"):
        self.normalize = normalize
        self.solver = solver

        # Will be set after fit
        self.X0 = None
        self.a0 = None
        self.fitted_ = False


    def _solve_ot(self, C, a0, a1):
        N0 = len(a0)
        N1 = len(a1)

        # Linear variable track expanded to a 2D row vector: Shape (1, N0 * N1)
        linear_indices = np.arange(N0 * N1)[None, :]

        # Row constraints: Map matching source indices. Shape: (N0, N0 * N1)
        row_mask = (np.arange(N0)[:, None] == linear_indices // N1)
        # Column constraints: Map matching target indices. Shape: (N1, N0 * N1)
        col_mask = (np.arange(N1)[:, None] == linear_indices % N1)

        # Total shape will be (N0 + N1, N0 * N1)
        dense_boolean_stack = np.vstack([row_mask, col_mask])
        A_eq = csr_matrix(dense_boolean_stack, dtype=np.float64)

        # A_eq = np.zeros((N0 + N1, N0 * N1))

        # # Row constraints (source)
        # for i in range(N0): A_eq[i, i * N1:(i + 1) * N1] = 1
        # # Column constraints (target)
        # for j in range(N1): A_eq[N0 + j, j::N1] = 1

        b_eq = np.concatenate([a0, a1])

        res = linprog(
            C.flatten(),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method=self.solver
        )

        if not res.success: raise ValueError(f"OT solver failed: {res.message}")
        return res.x.reshape(N0, N1)

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, X0, a0):
        """
        Store the reference measure.

        Parameters
        ----------
        X0 : array (N0, d)
        a0 : array (N0,)
        """
        X0, a0 = _prep_inputs(X0, a0, self.normalize)

        self.X0 = X0
        self.a0 = a0
        self.N0, self.d = X0.shape
        self.fitted_ = True

        return self

    def transform(self, X1, a1):
        """
        Compute LOT embedding of target measure.

        Parameters
        ----------
        X1 : array (N1, d)
        a1 : array (N1,)

        Returns
        -------
        s1_hat : array (N0, d)
        a1_hat : array (N0,)
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        X1, a1 = _prep_inputs(X1, a1, self.normalize)

        # Cost matrix
        C = _compute_cost(self.X0, X1)

        # Optimal transport plan
        Gamma = self._solve_ot(C, self.a0, a1)

        # LOT map
        a0_inv = np.diag(1.0 / self.a0)
        s1_hat = a0_inv @ (Gamma @ X1)

        # Mass is preserved on reference
        a1_hat = self.a0.copy()

        return s1_hat, a1_hat

    def fit_transform(self, X0, a0, X1, a1):
        """
        Convenience method: fit reference and transform target.
        """
        return self.fit(X0, a0).transform(X1, a1)