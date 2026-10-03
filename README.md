# jaxbessel

Differentiable Bessel functions of the first kind, $J_n(x)$, in JAX. It depends
only on `jax` and `numpy`.

```python
from jaxbessel import bessel_jn, j0, j1

bessel_jn(4, x)  # J_0(x) ... J_4(x), stacked along the first axis
```

- `j0` and `j1` follow the [CEPHES](https://www.netlib.org/cephes/) rational
  approximations.
- `bessel_jn(n, x)` returns every order up to `n`. It uses upward recurrence
  where that is stable ($|x| \ge n + 2$) and a folded trapezoidal sum of
  Bessel's integral ([arXiv:2206.05334](https://arxiv.org/abs/2206.05334))
  below that. Both agree with `scipy.special.jv` to about 1e-14 in float64.
- Derivatives use custom JVP rules from the recurrence identities
  $J_0' = -J_1$ and $J_m' = (J_{m-1} - J_{m+1})/2$, so `jax.grad`,
  `jax.jacfwd`, `jax.jacrev` and `jax.hessian` work to any order and are finite
  everywhere, including at $x = 0$.
- Works with and without `jax_enable_x64`: outputs follow the argument's dtype.

Used by [drpangloss](https://github.com/benjaminpope/drpangloss) and
[harmonix](https://github.com/shashankdholakia/harmonix). The JAX translation
of CEPHES `j0`/`j1` comes from Shashank Dholakia's harmonix.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
