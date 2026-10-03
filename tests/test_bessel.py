import ast
from pathlib import Path

import jax
import numpy as onp
import pytest
import jax.numpy as np
from jax.test_util import check_grads
from scipy.special import jv, jvp

import jaxbessel
from jaxbessel import bessel_jn, j0, j1

# jax.enable_x64 is newer than the oldest supported JAX, which only has the
# experimental context manager.
if hasattr(jax, "enable_x64"):
    enable_x64 = jax.enable_x64
else:
    from jax.experimental import enable_x64


@pytest.fixture(autouse=True)
def _float64_by_default():
    """Run these tests in float64, without changing the rest of the suite.

    Setting ``jax_enable_x64`` globally at import time made the precision of
    every other test module depend on collection order.
    """
    with enable_x64(True):
        yield


# (jax_enable_x64, expected dtype, absolute tolerance vs scipy)
BESSEL_PRECISIONS = [
    pytest.param(False, "float32", 2e-6, id="float32"),
    pytest.param(True, "float64", 1e-14, id="float64"),
]


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_bessel_jn_low_orders_match_scipy(order, x64, dtype, atol):
    # Covers x = 0, negative x, the small-x regime where the upward
    # recurrence alone is unstable, and the switch to the recurrence.
    xs = onp.linspace(-30.0, 30.0, 1201)
    expected = onp.array([jv(m, xs) for m in range(order + 1)])
    with enable_x64(x64):
        result = bessel_jn(order, np.asarray(xs))
    assert result.dtype == dtype
    assert result.shape == (order + 1, xs.size)
    assert onp.allclose(onp.asarray(result), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_bessel_jn_low_order_gradients_match_scipy(order, x64, dtype, atol):
    # d/dx J_n = (J_{n-1} - J_{n+1}) / 2, with J_{-1} = -J_1.
    xs = onp.linspace(-30.0, 30.0, 1201)
    expected = 0.5 * (jv(order - 1, xs) - jv(order + 1, xs))
    with enable_x64(x64):
        grad = jax.vmap(jax.grad(lambda z: bessel_jn(order, z)[order]))(
            np.asarray(xs)
        )
    assert grad.dtype == dtype
    assert onp.allclose(onp.asarray(grad), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [5, 8, 16])
def test_bessel_jn_high_orders_match_scipy(order):
    xs = onp.linspace(-60.0, 60.0, 2401)
    expected = onp.array([jv(m, xs) for m in range(order + 1)])
    result = bessel_jn(order, np.asarray(xs))
    assert onp.allclose(onp.asarray(result), expected, rtol=0.0, atol=1e-14)


def test_bessel_jn_accepts_scalars():
    assert bessel_jn(3, 2.5).shape == (4,)
    assert onp.allclose(bessel_jn(3, 2.5), [jv(m, 2.5) for m in range(4)])


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
def test_bessel_integer_inputs_are_promoted_to_float(x64, dtype, atol):
    # The coefficient tables are cast to the argument's dtype, so integer
    # arguments must reach them already promoted to float.
    xs = onp.arange(-12, 13)
    with enable_x64(x64):
        x = np.asarray(xs)
        assert onp.allclose(j0(x), jv(0, xs), rtol=0, atol=atol)
        assert onp.allclose(j1(x), jv(1, xs), rtol=0, atol=atol)
        result = bessel_jn(4, x)
    assert result.dtype == dtype
    expected = [jv(m, xs) for m in range(5)]
    assert onp.allclose(result, expected, rtol=0, atol=atol)


# Include x = 0, the CEPHES switch at |x| = 5 and the trig/recurrence switch at
# |x| = n + 2 for the orders tested.
DERIV_XS = onp.unique(
    onp.concatenate(
        [
            onp.linspace(-30.0, 30.0, 1201),
            [0.0, 4.999, 5.001, -5.001, 5.999, 6.001, 9.999, 10.001],
        ]
    )
)


@pytest.mark.parametrize("x64, dtype, atol", BESSEL_PRECISIONS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 8])
def test_bessel_jn_second_derivatives_match_scipy(order, x64, dtype, atol):
    with enable_x64(x64):
        second = jax.vmap(
            jax.grad(jax.grad(lambda z: bessel_jn(order, z)[order]))
        )(np.asarray(DERIV_XS))
    expected = jvp(order, DERIV_XS, 2)
    assert second.dtype == dtype
    assert onp.allclose(onp.asarray(second), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("order", [1, 4, 8])
def test_bessel_jn_jacobians_all_orders_match_scipy(order):
    xs = np.asarray(DERIV_XS)
    expected = onp.array([jvp(m, DERIV_XS, 1) for m in range(order + 1)])
    for jac in (jax.jacfwd, jax.jacrev):
        # Elementwise in x, so vmapping scalar Jacobians avoids materializing
        # the dense diagonal Jacobian over the x axes.
        diag = onp.asarray(jax.vmap(jac(lambda z: bessel_jn(order, z)))(xs)).T
        assert onp.allclose(diag, expected, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("x", [0.0, 0.3, 4.999, 5.001, -7.5, 9.999, 10.001])
def test_bessel_custom_rules_pass_check_grads(x):
    x = np.asarray(x)
    check_grads(j0, (x,), order=2, modes=("fwd", "rev"))
    check_grads(j1, (x,), order=2, modes=("fwd", "rev"))
    check_grads(lambda z: bessel_jn(8, z), (x,), order=2, modes=("fwd", "rev"))


def test_bessel_derivatives_at_zero_are_exact():
    zero = np.asarray(0.0)
    assert float(jax.grad(j1)(zero)) == pytest.approx(0.5, abs=1e-15)
    assert float(jax.grad(jax.grad(j0))(zero)) == pytest.approx(
        -0.5, abs=1e-15
    )
    assert float(jax.grad(jax.grad(j1))(zero)) == pytest.approx(0.0, abs=1e-15)
    hess = jax.jacfwd(jax.jacrev(lambda z: bessel_jn(4, z)))(zero)
    assert onp.all(onp.isfinite(onp.asarray(hess)))
    assert onp.allclose(onp.asarray(hess), [jvp(m, 0.0, 2) for m in range(5)])


def test_depends_only_on_jax_and_numpy():
    # Projects that share this package should not inherit anything else.
    tree = ast.parse((Path(jaxbessel.__file__).parent / "bessel.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert roots <= {"functools", "jax", "numpy"}, roots
