import jax
import numpy as onp
import pytest
import jax.numpy as np
from jax.test_util import check_grads
from scipy.special import spherical_jn

from jaxbessel import spherical_bessel_jn

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


# (jax_enable_x64, expected dtype, absolute tolerance vs scipy)
PRECISIONS = [
    pytest.param(False, "float32", 2e-6, id="float32"),
    pytest.param(True, "float64", 1e-14, id="float64"),
]

# Include x = 0, negative x and both sides of the quadrature/recurrence switch
# at |x| = n + 2 for the orders tested.
XS = onp.unique(
    onp.concatenate(
        [
            onp.linspace(-60.0, 60.0, 2401),
            [0.0, 1.999, 2.001, 3.999, 4.001, -6.001, 9.999, 10.001, 18.001],
        ]
    )
)


def _scipy_jn(m, xs, derivative=False):
    # Older scipy returns nan for x < 0, so use j_m(-x) = (-1)^m j_m(x).
    xs = onp.asarray(xs, dtype=float)
    parity = onp.where(xs < 0, (-1.0) ** (m + derivative), 1.0)
    return parity * spherical_jn(m, onp.abs(xs), derivative=derivative)


def _reference(order, xs, derivative=False):
    return onp.array([_scipy_jn(m, xs, derivative) for m in range(order + 1)])


@pytest.mark.parametrize("x64, dtype, atol", PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 8, 16])
def test_spherical_bessel_jn_matches_scipy(order, x64, dtype, atol):
    with enable_x64(x64):
        result = spherical_bessel_jn(order, np.asarray(XS))
    assert result.dtype == dtype
    assert result.shape == (order + 1, XS.size)
    assert onp.allclose(onp.asarray(result), _reference(order, XS), rtol=0, atol=atol)


@pytest.mark.parametrize("x64, dtype, atol", PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 8])
def test_spherical_bessel_jn_gradients_match_scipy(order, x64, dtype, atol):
    with enable_x64(x64):
        grad = jax.vmap(jax.grad(lambda z: spherical_bessel_jn(order, z)[order]))(
            np.asarray(XS)
        )
    expected = _scipy_jn(order, XS, derivative=True)
    assert grad.dtype == dtype
    assert onp.allclose(onp.asarray(grad), expected, rtol=0, atol=atol)


@pytest.mark.parametrize("order", [1, 4, 8])
def test_spherical_bessel_jn_jacobians_all_orders_match_scipy(order):
    expected = _reference(order, XS, derivative=True)
    for jac in (jax.jacfwd, jax.jacrev):
        diag = jax.vmap(jac(lambda z: spherical_bessel_jn(order, z)))(np.asarray(XS))
        assert onp.allclose(onp.asarray(diag).T, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("order", [0, 3, 8])
def test_spherical_bessel_jn_second_derivatives(order):
    # j_m'' = -(2/x) j_m' - (1 - m(m+1)/x^2) j_m, away from x = 0.
    xs = XS[onp.abs(XS) > 0.5]
    second = jax.vmap(jax.grad(jax.grad(lambda z: spherical_bessel_jn(order, z)[order])))(
        np.asarray(xs)
    )
    j = _scipy_jn(order, xs)
    dj = _scipy_jn(order, xs, derivative=True)
    expected = -2 / xs * dj - (1 - order * (order + 1) / xs**2) * j
    assert onp.allclose(onp.asarray(second), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("x", [0.0, 0.3, 1.999, 2.001, -7.5, 9.999, 10.001])
def test_spherical_custom_rule_passes_check_grads(x):
    check_grads(
        lambda z: spherical_bessel_jn(8, z), (np.asarray(x),), order=2, modes=("fwd", "rev")
    )


def test_spherical_derivatives_at_zero_are_exact():
    zero = np.asarray(0.0)
    # j_1'(0) = 1/3 and j_0''(0) = -1/3; every other first and second
    # derivative at zero vanishes except j_2''(0) = 2/15.
    jac = jax.jacfwd(lambda z: spherical_bessel_jn(4, z))(zero)
    hess = jax.jacfwd(jax.jacrev(lambda z: spherical_bessel_jn(4, z)))(zero)
    assert onp.allclose(onp.asarray(jac), [0, 1 / 3, 0, 0, 0], rtol=0, atol=1e-15)
    assert onp.allclose(
        onp.asarray(hess), [-1 / 3, 0, 2 / 15, 0, 0], rtol=0, atol=1e-15
    )


def test_spherical_bessel_jn_accepts_scalars_and_integers():
    assert spherical_bessel_jn(3, 2.5).shape == (4,)
    assert onp.allclose(spherical_bessel_jn(3, 2.5), _reference(3, 2.5))
    xs = onp.arange(-12, 13)
    assert onp.allclose(spherical_bessel_jn(4, np.asarray(xs)), _reference(4, xs), atol=1e-14)


def test_spherical_bessel_jn_broadcasts_over_arrays():
    x = np.asarray(XS[:600].reshape(20, 30))
    result = spherical_bessel_jn(5, x)
    assert result.shape == (6, 20, 30)
    grad = jax.grad(lambda z: spherical_bessel_jn(5, z).sum())(x)
    assert grad.shape == x.shape
