import jax
import numpy as onp
import pytest
import jax.numpy as np
from jax.test_util import check_grads
from scipy.special import gamma, jv

from jaxbessel import bessel_jn, bessel_jv_over_xv, spherical_bessel_jn

# jax.enable_x64 is newer than the oldest supported JAX, which only has the
# experimental context manager.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    from jax.experimental import enable_x64


@pytest.fixture(autouse=True)
def _float64_by_default():
    with enable_x64(True):
        yield


# (jax_enable_x64, expected dtype, tolerance relative to the envelope). In
# float32 the phase of cos(x) alone is uncertain by x * 6e-8, about 1e-5 at
# x = 200.
PRECISIONS = [
    pytest.param(False, "float32", 5e-5, id="float32"),
    pytest.param(True, "float64", 2e-11, id="float64"),
]

# Include x = 0, negative x and both sides of the switch at |x| = 14.
XS = onp.unique(
    onp.concatenate(
        [
            onp.linspace(-200.0, 200.0, 4001),
            [0.0, 1e-3, 13.999, 14.0, 14.001, -14.001],
        ]
    )
)
ORDERS = [-0.45, -0.25, 0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.75, 3.5, 6.0, 11.0, 12.0]


def _reference(nu, xs):
    ax = onp.abs(onp.asarray(xs, dtype=float))
    at_zero = 1 / (2**nu * gamma(nu + 1))
    safe = onp.where(ax == 0, 1.0, ax)
    return onp.where(ax == 0, at_zero, jv(nu, safe) / safe**nu)


def _envelope(nu, xs):
    # 1 / (2^nu Gamma(nu + 1)) at small x, the asymptotic amplitude at large x
    ax = onp.maximum(onp.abs(onp.asarray(xs, dtype=float)), 1e-300)
    at_zero = 1 / (2**nu * gamma(nu + 1))
    with onp.errstate(over="ignore"):
        return onp.minimum(at_zero, onp.sqrt(2 / (onp.pi * ax)) * ax**-nu)


@pytest.mark.parametrize("x64, dtype, rtol", PRECISIONS)
@pytest.mark.parametrize("nu", ORDERS)
def test_matches_scipy(nu, x64, dtype, rtol):
    with enable_x64(x64):
        result = bessel_jv_over_xv(nu, np.asarray(XS))
    assert result.dtype == dtype
    assert result.shape == XS.shape
    err = onp.abs(onp.asarray(result, dtype=float) - _reference(nu, XS))
    assert onp.all(err <= rtol * _envelope(nu, XS))


@pytest.mark.parametrize("nu", ORDERS)
def test_value_at_zero(nu):
    assert onp.isclose(
        bessel_jv_over_xv(nu, 0.0), 1 / (2**nu * gamma(nu + 1)), rtol=1e-14
    )


def test_matches_integer_and_spherical_orders():
    # J_n(x) / x^n from bessel_jn, and J_{l+1/2}(x) / x^{l+1/2} from
    # j_l(x) = sqrt(pi / (2x)) J_{l+1/2}(x); dividing by x^n near zero would
    # amplify their rounding, so stay away from it
    xs = XS[onp.abs(XS) > 1]
    j = onp.asarray(bessel_jn(3, np.asarray(xs)))
    sph = onp.asarray(spherical_bessel_jn(3, np.asarray(xs)))
    for n in range(4):
        expected = j[n] / xs**n
        assert onp.allclose(
            bessel_jv_over_xv(n, xs), expected, rtol=0, atol=2e-12 * _envelope(n, xs)
        )
        nu = n + 0.5
        expected = sph[n] * onp.sqrt(2 / onp.pi) / xs**n
        assert onp.allclose(
            bessel_jv_over_xv(nu, xs), expected, rtol=0, atol=2e-12 * _envelope(nu, xs)
        )


@pytest.mark.parametrize("x64, dtype, rtol", PRECISIONS)
@pytest.mark.parametrize("nu", [-0.25, 0.0, 1.25, 2.0, 3.5, 12.0])
def test_gradients_match_identity(nu, x64, dtype, rtol):
    # d/dx [J_nu(x) / x^nu] = -x J_{nu+1}(x) / x^{nu+1}
    with enable_x64(x64):
        grad = jax.vmap(jax.grad(lambda z: bessel_jv_over_xv(nu, z)))(np.asarray(XS))
    expected = -XS * _reference(nu + 1, XS)
    assert grad.dtype == dtype
    err = onp.abs(onp.asarray(grad, dtype=float) - expected)
    assert onp.all(err <= rtol * onp.abs(XS) * _envelope(nu + 1, XS) + 1e-300)


@pytest.mark.parametrize("x", [0.0, 0.3, 5.0, 13.999, 14.001, -30.0])
def test_custom_rule_passes_check_grads(x):
    check_grads(
        lambda z: bessel_jv_over_xv(1.25, z), (np.asarray(x),), order=2, modes=("fwd", "rev")
    )


def test_derivatives_at_zero():
    # f(x) = c (1 - x^2 / (4 (nu + 1)) + ...), so f'(0) = 0 and
    # f''(0) = -c / (2 (nu + 1)) with c = 1 / (2^nu Gamma(nu + 1))
    nu = 1.25
    c = 1 / (2**nu * gamma(nu + 1))
    zero = np.asarray(0.0)
    f = lambda z: bessel_jv_over_xv(nu, z)  # noqa: E731
    assert jax.grad(f)(zero) == 0.0
    assert onp.isclose(jax.grad(jax.grad(f))(zero), -c / (2 * (nu + 1)), rtol=1e-13)


def test_broadcasts_and_accepts_scalars():
    x = np.asarray(XS[:600].reshape(20, 30))
    assert bessel_jv_over_xv(1.25, x).shape == (20, 30)
    assert bessel_jv_over_xv(1.25, 2.5).shape == ()
    assert jax.grad(lambda z: bessel_jv_over_xv(1.25, z).sum())(x).shape == x.shape


@pytest.mark.parametrize("nu", [-0.5, -1.0, 12.5, 172.0])
def test_rejects_orders_outside_the_supported_range(nu):
    with pytest.raises(ValueError):
        bessel_jv_over_xv(nu, 1.0)
