# jaxbessel

Differentiable Bessel functions of the first kind in JAX: cylindrical $J_n(x)$
and spherical $j_n(x)$. It depends only on `jax` and `numpy`.

```bash
pip install jaxbessel
```

```python
from jaxbessel import bessel_jn, j0, j1, spherical_bessel_jn

bessel_jn(4, x)            # J_0(x) ... J_4(x), stacked along the first axis
spherical_bessel_jn(4, x)  # j_0(x) ... j_4(x), likewise
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
- Derivatives use custom JVP rules from the recurrence identities
  ($J_m' = (J_{m-1} - J_{m+1})/2$ and
  $j_m' = (m\,j_{m-1} - (m+1)\,j_{m+1})/(2m+1)$), so `jax.grad`, `jax.jacfwd`,
  `jax.jacrev` and `jax.hessian` work to any order and are finite everywhere,
  including at $x = 0$.
- Works with and without `jax_enable_x64`: outputs follow the argument's dtype.

Used by [drpangloss](https://github.com/benjaminpope/drpangloss) and
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
