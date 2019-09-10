"""Borrowed from trmf and re-purposed for torch and product spaces."""
import torch
from math import sqrt


def clone(a):
    return [u.clone() for u in a]


def dzero(a):
    return [u.zero_() for u in a]


def ddot(a, b):
    return sum(u.flatten() @ v.flatten() for u, v in zip(a, b))


def daxpy(alpha, x, y):
    return [b.add_(alpha * a) for a, b in zip(x, y)]


def dscal(alpha, x):
    return [a.mul_(alpha) for a in x]


def dflat(x):
    return torch.cat([u.flatten() for u in x])


def dnorm(x):
    return torch.norm(dflat(x), dim=-1)


def trcg(Ax, r, x, n_iterations=10000, tr_delta=0, rtol=1e-5, atol=1e-8,
         args=(), verbose=False):
    r"""Simple Conjugate gradient solver with trust region control.

    Approximately solves `r = A(z - x)` w.r.t. `z` and updating `r` and `x`
    inplace with the final residual and solution `z`, respectively.

    Details
    -------
    For the given `r` this procedure uses conjugate gradients method to solve
    the least squares problem within the trust region of radius :math:`\delta`
    around `x`:

        .. math ::
            \min_{p} \|A p - r \|^2
            s.t. \|x + p\| \leq \delta

    and returns `z = x + p` as the solution. The residual `r` and the point `x`
    are updated inplace with the final residual and solution `z`, respectively,
    upon termination.

    Backtracking
    ------------
    In contrast to the implementation on LIBLINEAR, this version backtrack into
    the trust region upon a breach event. It does using the analytic solution
    to the following 1-dimensional optimization problem::

        .. math ::
            \min_{\eta \geq 0} \eta
            s.t. \| (z - x) - \eta p \| \leq \delta

    where :math:`\|z - x\| > \delta` and `p` is the current conjugate
    minimizing direction. The solution is given by

        .. math ::
            \eta = \frac1{\|p\|} (q - \sqrt{q^2 + \delta^2 - \|z-x\|^2})

    where :math:`q = \frac{p'(z-x)}{\|p\|}`.

    Arguments
    ---------
    Ax : callable
        The function with declaration `f(x, *args)` that computes the matrix
        vector product for the given point `x`.

    r : flat writable numpy array
        The initial residual vector to solve the linear system for. The array
        in `r` is updated INPLACE during the iterations. The final solution
        residual is returned implicitly in `r`.

    x : flat writable numpy array
        The initial point and final solution of the linear system. The array
        in `x` is updated INPLACE during the iterations. The solution is
        returned implicitly in `x`.

    n_iterations : int, optional (default=1000)
        The number of tron iterations.

    tr_delta : double, optional (default=0)
        The radius of the trust region around the initial point in `x`. The
        conjugate gradient steps steps are terminated if the trust region is
        breached, in which case the constraint violating step is retracted
        back to the trust region boundary.

    rtol : double, optional (default=1e-3)
        The relative reduction of the l2 norm of the gradient, to be used in
        for convergence criterion. The default set to match the corresponding
        setting in LIBLINEAR.

    atol : double, optional (default=1e-5)
        The minimal absolute reduction in the l2 norm of the gradient.

    args : tuple, optional (default=empty tuple)
        The extra positional arguments to pass the the callables in `func`.

    verbose : bool, optional (default=False)
        Whether to print debug diagnostics regarding the convergence, the
        current trust region radius, gradients, CG iterations and step sizes.

    Examples
    --------
    Import numpy and trcg from this library

    >>> import numpy as np
    >>> from trmf.tron import trcg

    Create random PSD matrix A

    >>> A = np.random.normal(scale=0.1, size=(10000, 200))
    >>> A = np.dot(A.T, A)

    Solve using linalg.inv (pivoting)

    >>> r_0, a_0 = np.ones(200), np.ones(200)
    >>> z_0 = a_0 + np.linalg.solve(A, r_0)

    Solve using `trcg`

    >>> r, z = r_0.copy(), a_0.copy()
    >>> trcg(lambda p: np.dot(A, p), r, z, verbose=False)
    >>> assert np.allclose(z, z_0)
    """
    if n_iterations > 0:
        n_iterations = min(n_iterations, sum(u.numel() for u in x))

    p, iteration = clone(r), 0
    tr_delta_sq = tr_delta ** 2

    rtr, rtr_old = float(ddot(r, r)), 1.0
    cg_tol = sqrt(rtr) * rtol + atol
    region_breached = False
    while (iteration < n_iterations) and (sqrt(rtr) > cg_tol):
        Ap = Ax(p, *args)
        iteration += 1
        if verbose:
            print("""iter %2d |Ap| %5.3e |p| %5.3e """
                  """|r| %5.3e |x| %5.3e beta %5.3e""" %
                  (iteration, dnorm(Ap), dnorm(p),
                   dnorm(r), dnorm(x), rtr / rtr_old))
        # end if

        alpha = rtr / float(ddot(p, Ap))
        daxpy(alpha, p, x)  # x += alpha * p
        daxpy(-alpha, Ap, r)  # r -= alpha * Ap

        # check trust region (diverges from `tron.cpp` in liblinear and leml-imf)
        if tr_delta_sq > 0:
            xTx = float(ddot(x, x))
            if xTx > tr_delta_sq:
                xTp = float(ddot(x, p))
                if xTp > 0:
                    # backtrack into the trust region
                    p_nrm = dnorm(p)

                    q = xTp / p_nrm
                    eta = (q - sqrt(max(q * q + tr_delta_sq - xTx, 0))) / p_nrm

                    # re-project onto the boundary of the region
                    daxpy(eta, Ap, r)  # r += eta * Ap
                    daxpy(-eta, p, x)  # x -= eta * p
                else:
                    # this never happens maybe due to CG iteration properties
                    pass
                # end if

                region_breached = True
                break
            # end if
        # end if

        rtr, rtr_old = float(ddot(r, r)), rtr
        dscal(rtr / rtr_old, p)  # p *= rtr / rtr_old
        daxpy(1., r, p)  # p += r

    # end while

    return iteration, region_breached