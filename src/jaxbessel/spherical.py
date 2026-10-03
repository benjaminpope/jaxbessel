r"""Spherical Bessel functions of the first kind, j_n(x), in JAX.

These replace harmonix's ``csphjy`` (a JAX port of Zhang & Jin's backward
recurrence) with the same layout as ``bessel_jn``: ``spherical_bessel_jn``
returns all orders up to ``n``. For ``|x| < n + 2``, where upward recurrence
is unstable, it uses Gauss-Legendre quadrature of
$j_m(x) = \frac{(-i)^m}{2}\int_{-1}^{1} e^{ixt} P_m(t)\,dt$, and the
recurrence above it. Values and gradients are finite everywhere, including at
``x = 0``, and real-valued.

Derivatives use a custom JVP rule from the recurrence identity
$j_m' = (m\,j_{m-1} - (m + 1)\,j_{m+1}) / (2m + 1)$, so the $k$-th derivative
of ``spherical_bessel_jn(n, x)`` costs one call to
``spherical_bessel_jn(n + k, x)``.
"""

from functools import partial

import jax
import jax.numpy as np
import numpy as onp
from jax import jit
from jax.lax import scan


__all__ = ["spherical_bessel_jn"]


def _spherical_jn_quad(n, x, nodes):
    r"""All orders ``0..n`` of $j_m(x)$ from ``nodes``-point Gauss-Legendre quadrature.

    $P_m(t)$ has the parity of $m$, so folding the integral onto $t > 0$
    leaves $j_m(x) = (-1)^{m/2} \int_0^1 \cos(xt) P_m(t)\,dt$ for even $m$ and
    $(-1)^{(m-1)/2} \int_0^1 \sin(xt) P_m(t)\,dt$ for odd $m$. The integrand is
    a degree-$m$ polynomial times $e^{ixt}$, which the quadrature integrates to
    machine precision once ``nodes`` comfortably exceeds $m + e|x|/2$.

    ``nodes`` must be even, so that there are ``nodes / 2`` positive nodes and
    none at zero; the weights are built at trace time.
    """
    t, w = onp.polynomial.legendre.leggauss(nodes)
    t, w = t[nodes // 2 :], w[nodes // 2 :]
    orders = onp.arange(n + 1)[:, None]
    signed = (-1.0) ** (orders // 2) * w * onp.polynomial.legendre.legvander(t, n).T
    even = orders % 2 == 0
    weights = np.asarray(
        onp.concatenate([onp.where(even, signed, 0.0), onp.where(even, 0.0, signed)], axis=1),
        dtype=x.dtype,
    )

    arg = x[..., None] * np.asarray(t, dtype=x.dtype)
    trig = np.concatenate([np.cos(arg), np.sin(arg)], axis=-1)
    return np.moveaxis(trig @ weights.T, -1, 0)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _spherical_bessel_jn(n, x):
    """All orders ``0..n`` of $j_m(x)$ for a float array ``x``; see ``spherical_bessel_jn``."""
    # Upward recurrence is accurate for |x| > n. With n + n // 4 + 20 nodes
    # (rounded up to even) the quadrature is accurate to machine precision up
    # to x_switch for every order <= n.
    x_switch = float(n + 2)
    nodes = 2 * -(-(n + n // 4 + 20) // 2)
    small = np.abs(x) < x_switch
    x_rec = np.where(small, x_switch, x)

    def body(carry, m):
        jm_minus, jm = carry
        jm_plus = (2 * m + 1) / x_rec * jm - jm_minus
        return (jm, jm_plus), jm_plus

    sin, cos = np.sin(x_rec), np.cos(x_rec)
    j0_rec = sin / x_rec
    j1_rec = (j0_rec - cos) / x_rec
    _, j_high = scan(body, (j0_rec, j1_rec), np.arange(1, n))
    j_rec = np.concatenate([j0_rec[None], j1_rec[None], j_high])[: n + 1]
    j_quad = _spherical_jn_quad(n, np.where(small, x, 0.0), nodes)
    return np.where(small, j_quad, j_rec)


@_spherical_bessel_jn.defjvp
def _spherical_bessel_jn_jvp(n, primals, tangents):
    (x,), (x_dot,) = primals, tangents
    j = _spherical_bessel_jn(n + 1, x)
    # j_m' = (m j_{m-1} - (m + 1) j_{m+1}) / (2m + 1), so j_0' = -j_1. As in
    # bessel_jn, the primal comes from the same order-(n + 1) evaluation.
    m = np.arange(n + 1, dtype=x.dtype).reshape((n + 1,) + (1,) * x.ndim)
    j_below = np.concatenate([np.zeros_like(j[:1]), j[:n]])
    deriv = (m * j_below - (m + 1) * j[1 : n + 2]) / (2 * m + 1)
    return j[: n + 1], deriv * x_dot


@partial(jit, static_argnums=0)
def spherical_bessel_jn(n, x):
    r"""Compute the spherical Bessel function $j_n(x)$, for $n >= 0$. Returns the
    function output for all orders up to the requested order $n$ evaluated for the
    kernel $x$, stacked along the first axis, so the shape of the result is
    (n + 1, shape(x)).

    All orders come from the upward recurrence
    $j_{m+1} = ((2m + 1)/x) j_m - j_{m-1}$, seeded by $j_0 = \sin x / x$ and
    $j_1 = (j_0 - \cos x) / x$, where it is stable ($|x| \ge n + 2$), and from
    Gauss-Legendre quadrature (see ``_spherical_jn_quad``) below that. Both
    agree with ``scipy.special.spherical_jn`` to about 1e-14, and the
    gradients are finite everywhere, including $x = 0$.

    Derivatives of all orders use
    $j_m' = (m\,j_{m-1} - (m + 1)\,j_{m+1}) / (2m + 1)$, evaluated from
    ``spherical_bessel_jn(n + 1, x)``, rather than differentiating through the
    approximations.
    """
    return _spherical_bessel_jn(n, np.asarray(x, dtype=float))
