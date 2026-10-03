r"""$J_\nu(x) / x^\nu$ for any real order $\nu > -1/2$, in JAX.

This is the kernel of the visibility of a disk whose brightness is a power of
$\mu = \sqrt{1 - r^2}$: a term $\mu^a$ transforms to
$J_{a/2+1}(x) / x^{a/2+1}$ (Quirrenbach et al. 1996, A&A 312, 160, eq. 3).
Polynomial limb darkening only needs integer and half-integer orders, which
``bessel_jn`` and ``spherical_bessel_jn`` already give, but laws with
$\sqrt{\mu}$ terms need orders such as $5/4$.

For $|x| < 14$ it uses Poisson's integral,
$J_\nu(x) / x^\nu = \frac{1}{2^\nu \sqrt{\pi}\,\Gamma(\nu + 1/2)}
\int_{-1}^{1} (1 - t^2)^{\nu - 1/2} \cos(xt)\,dt$, by Gauss-Gegenbauer
quadrature. It is regular at $x = 0$, where it equals
$1 / (2^\nu \Gamma(\nu + 1))$. Above that it uses Hankel's asymptotic
expansion, which loses accuracy below $|x| \approx 14$, just as the quadrature
loses digits to cancellation above it.

Derivatives use a custom JVP rule from the identity
$\frac{d}{dx}[J_\nu(x)/x^\nu] = -x\,J_{\nu+1}(x)/x^{\nu+1}$, so the $k$-th
derivative costs one evaluation at order $\nu + k$.
"""

import math
from functools import partial

import jax
import jax.numpy as np
import numpy as onp
from jax import jit


__all__ = ["bessel_jv_over_xv"]

X_SWITCH = 14.0
NODES = 64  # quadrature nodes; 32 per side of t = 0
TERMS = 14  # pairs of terms in the asymptotic expansion


def _gegenbauer_rule(nu, nodes):
    r"""Gauss rule for the weight $(1 - t^2)^{\nu - 1/2}$ on $[-1, 1]$ (Golub-Welsch)."""
    k = onp.arange(2, nodes)
    off = onp.sqrt(k * (k + 2 * nu - 1) / (4 * (k + nu) * (k + nu - 1)))
    # the k = 1 term, simplified so that it stays finite at nu = 0
    off = onp.concatenate([[onp.sqrt(0.5 / (1 + nu))], off])
    t, vectors = onp.linalg.eigh(onp.diag(off, 1) + onp.diag(off, -1))
    mass = math.sqrt(math.pi) * math.exp(
        math.lgamma(nu + 0.5) - math.lgamma(nu + 1)
    )
    return t, mass * vectors[0] ** 2


def _quadrature(nu, x):
    t, w = _gegenbauer_rule(nu, NODES)
    # the rule is symmetric: fold onto the positive nodes
    t, w = t[NODES // 2 :], 2.0 * w[NODES // 2 :]
    w = w / (2**nu * math.sqrt(math.pi) * math.gamma(nu + 0.5))
    arg = x[..., None] * np.asarray(t, dtype=x.dtype)
    return np.cos(arg) @ np.asarray(w, dtype=x.dtype)


def _asymptotic(nu, x):
    r"""Hankel's expansion of $J_\nu(x) / x^\nu$ for large positive ``x``."""
    # a_k = (4 nu^2 - 1)(4 nu^2 - 9)...(4 nu^2 - (2k - 1)^2) / (k! 8^k)
    a = [1.0]
    for k in range(1, 2 * TERMS + 2):
        a.append(a[-1] * (4 * nu**2 - (2 * k - 1) ** 2) / (8 * k))
    y = 1.0 / x
    p = sum((-1) ** k * a[2 * k] * y ** (2 * k) for k in range(TERMS + 1))
    q = sum(
        (-1) ** k * a[2 * k + 1] * y ** (2 * k + 1) for k in range(TERMS + 1)
    )
    chi = x - (nu / 2 + 0.25) * math.pi
    amp = np.sqrt(2.0 / math.pi * y) * y**nu
    return amp * (p * np.cos(chi) - q * np.sin(chi))


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _bessel_jv_over_xv(nu, x):
    r"""$J_\nu(x) / x^\nu$ for a float array ``x``; see ``bessel_jv_over_xv``."""
    ax = np.abs(x)  # the function is even in x
    small = ax < X_SWITCH
    near = _quadrature(nu, np.where(small, ax, 0.0))
    far = _asymptotic(nu, np.where(small, X_SWITCH, ax))
    return np.where(small, near, far)


@_bessel_jv_over_xv.defjvp
def _bessel_jv_over_xv_jvp(nu, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return (
        _bessel_jv_over_xv(nu, x),
        -x * _bessel_jv_over_xv(nu + 1, x) * x_dot,
    )


@partial(jit, static_argnums=0)
def bessel_jv_over_xv(nu, x):
    r"""Compute $J_\nu(x) / x^\nu$ for a real order $\nu > -1/2$.

    ``nu`` is a static Python float; ``x`` is any float array, and the result
    has its shape. The function is even in $x$ and regular at $x = 0$, where
    it equals $1 / (2^\nu \Gamma(\nu + 1))$, with finite gradients there.

    Gauss-Gegenbauer quadrature of Poisson's integral is used for
    $|x| < 14$ and Hankel's asymptotic expansion above. For
    $0 \le \nu \le 11$ the result agrees with
    ``scipy.special.jv(nu, x) / x**nu`` to about 1e-12 of its envelope,
    $\min(1 / (2^\nu \Gamma(\nu + 1)), \sqrt{2 / (\pi x)}\,x^{-\nu})$, in
    float64, with the largest errors next to the switch.

    Derivatives of all orders use
    $\frac{d}{dx}[J_\nu(x)/x^\nu] = -x\,J_{\nu+1}(x)/x^{\nu+1}$ rather than
    differentiating through the approximations.
    """
    if not nu > -0.5:
        raise ValueError(f"bessel_jv_over_xv needs nu > -1/2, got {nu}.")
    return _bessel_jv_over_xv(float(nu), np.asarray(x, dtype=float))
