# jaxbessel

Differentiable Bessel functions of the first kind in JAX: cylindrical $J_n(x)$,
spherical $j_n(x)$, and $J_\nu(x)/x^\nu$ for any real order $\nu > -1/2$. It
depends only on `jax` and `numpy`.

```bash
pip install jaxbessel
```

```python
from jaxbessel import bessel_jn, bessel_jv_over_xv, j0, j1, spherical_bessel_jn

bessel_jn(4, x)               # J_0(x) ... J_4(x), stacked along the first axis
spherical_bessel_jn(4, x)     # j_0(x) ... j_4(x), likewise
bessel_jv_over_xv(1.25, x)    # J_{5/4}(x) / x^{5/4}, regular at x = 0
```

- `j0` and `j1` follow the [CEPHES](https://www.netlib.org/cephes/) rational
  approximations.
- `bessel_jn(n, x)` returns every order up to `n`. It uses upward recurrence
  where that is stable ($|x| \ge n + 2$) and a folded trapezoidal sum of
  Bessel's integral ([arXiv:2206.05334](https://arxiv.org/abs/2206.05334))
  below that.
- `spherical_bessel_jn(n, x)` does the same for $j_n$, with Gauss-Legendre
  quadrature of $j_m(x) = \frac{(-i)^m}{2}\int_{-1}^{1} e^{ixt} P_m(t)\,dt$
  below the switch.
- Both agree with `scipy.special` to about 1e-14 in float64.
- `bessel_jv_over_xv(nu, x)` is the kernel of the visibility of a stellar disk
  whose brightness is a power of $\mu$ (Quirrenbach et al. 1996, A&A 312, 160,
  eq. 3), needed for orders such as $5/4$ in the square-root limb-darkening
  law. It uses Gauss-Gegenbauer quadrature of Poisson's integral for
  $|x| < 14$ and Hankel's asymptotic expansion above, and agrees with
  `scipy.special.jv(nu, x) / x**nu` to about 1e-12 of its envelope in float64
  for $0 \le \nu \le 11$. Its derivative is
  $-x\,J_{\nu+1}(x)/x^{\nu+1}$.
- Derivatives use custom JVP rules from the recurrence identities
  ($J_m' = (J_{m-1} - J_{m+1})/2$,
  $j_m' = (m\,j_{m-1} - (m+1)\,j_{m+1})/(2m+1)$ and the one above), so `jax.grad`, `jax.jacfwd`,
  `jax.jacrev` and `jax.hessian` work to any order and are finite everywhere,
  including at $x = 0$.
- Works with and without `jax_enable_x64`: outputs follow the argument's dtype.

Used by [virgil](https://github.com/benjaminpope/virgil) and
[harmonix](https://github.com/shashankdholakia/harmonix). The JAX translation
of CEPHES `j0`/`j1`, and the original spherical Bessel functions this package
replaces, come from Shashank Dholakia's harmonix.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
