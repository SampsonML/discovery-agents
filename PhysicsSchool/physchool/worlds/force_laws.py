"""Pairwise force and potential laws for ``NBodySampler``.

Each callable here has the signature

    ``force_law(r_mag, q_i, q_j, m_i, m_j) -> F_mag``
    ``potential_law(r_mag, q_i, q_j, m_i, m_j) -> V``

where ``q_i, q_j`` are per-particle "charges" / source couplings (signed),
``m_i, m_j`` are inertias, and the simulator uses the convention

    Force on particle i from j  =  F_mag * (r_j - r_i) / |r_j - r_i|

so a *positive* ``F_mag`` corresponds to an *attractive* force.  Repulsive
forces fall out automatically when ``q_i * q_j < 0``.  The companion potential
is normalised so that ``F_mag = dV/dr`` (i.e. attractive → V increases with
separation; bound systems sit in the V<0 region for short-ranged kernels).

All implementations use ``jax.numpy`` so they JIT-compile cleanly.  The 2D
Yukawa kernel calls ``scipy.special.k0/k1`` via ``jax.pure_callback`` because
JAX has no native modified-Bessel functions; this still traces through JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

# ── 3D-style "Newtonian" gravity (uses charges as gravitational mass) ──────


def gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """Newtonian inverse-square attractive force, ``F = G q_i q_j / r^2``."""
    return G * q_i * q_j / r_mag**2


def gravity_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """Companion potential, ``V = -G q_i q_j / r``."""
    return -G * q_i * q_j / r_mag


def screened_gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 5.0):
    """3D-style exponentially screened inverse-square force.

    ``F = G q_i q_j exp(-r/lam) / r^2``.  Used by the screening
    stress-test; the correct *2D* screened-Poisson kernel lives in
    ``yukawa_2d_force`` and is the one that matches ``FieldSampler``.
    """
    return G * q_i * q_j / r_mag**2 * jnp.exp(-r_mag / lam)


# ── Coulomb (electrostatic) — physics-standard sign convention ────────────

# In real-world electrostatics (and in the codebase's pair convention where
# F_mag > 0 along (r_j - r_i) is attractive), Coulomb's law is
#     F = -k q_i q_j / r²,
# i.e. opposite-sign charges attract (q_i q_j < 0 → F_mag > 0) and like-sign
# charges repel (q_i q_j > 0 → F_mag < 0). This is the *opposite* sign from
# ``gravity_force`` above, which models same-sign-attractive Newtonian
# gravity (where "charges" are masses, always positive).


def coulomb_force(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    """Coulomb 1/r² force, ``F = -k q_i q_j / r²`` (opposite-sign attracts)."""
    return -k * q_i * q_j / r_mag**2


def coulomb_potential(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    """Companion potential, ``V = k q_i q_j / r`` (F_mag = dV/dr)."""
    return k * q_i * q_j / r_mag


def running_coupling_force(r_mag, q_i, q_j, m_i, m_j, g0: float = 0.5, r0: float = 5.0):
    """Logarithmically running coupling ``g(r) = g0 / (1 + g0 ln(r/r0))``.

    Used to mimic QCD-like asymptotic freedom: at ``r << r0`` the effective
    coupling grows; at ``r >> r0`` it decays logarithmically.
    """
    g = g0 / (1.0 + g0 * jnp.log(r_mag / r0))
    return g * q_i * q_j / r_mag**2


# ── 2D Poisson (Laplacian Green's function in 2D) ─────────────────────────

# In 2D, the Green's function of ∇² is logarithmic, G(r) = ln(r) / (2π).
# A point source of charge q at the origin produces field
#     φ(r) = q ln(r) / (2π)
# which exerts an attractive force of magnitude q_i q_j / (2π r) on a second
# particle of charge q_j at separation r (for like-signed charges).
# This is the kernel that ``FieldSampler`` with ``operators=[{type:laplacian}]``
# is computing on the grid.

_INV_TWO_PI = 1.0 / (2.0 * jnp.pi)


def poisson_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """2D Poisson force: ``F = G q_i q_j / (2π r)``."""
    return G * q_i * q_j * _INV_TWO_PI / r_mag


def poisson_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """2D Poisson potential, ``V = G q_i q_j ln(r) / (2π)``.

    The logarithm has no natural ``r → ∞`` reference, so ``V(r=1) = 0`` by
    construction.  Only differences in PE are physically meaningful.
    """
    return G * q_i * q_j * _INV_TWO_PI * jnp.log(r_mag)


# ── 2D Yukawa / screened Poisson (modified Helmholtz) ─────────────────────

# In 2D, the Green's function of (-∇² + 1/λ²) is
#     G(r) = K_0(r/λ) / (2π)
# where K_0 is the modified Bessel function of the second kind.  The force
# magnitude is then |dG/dr| = K_1(r/λ) / (2π λ).  We expose ``scipy.special``
# inside JAX via pure_callback (no native JAX modified-Bessel-K).


def _k0_jax(x):
    """K_0 for jnp arrays via scipy callback (pure → JIT-compatible)."""
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k0(xx), dtype=xx.dtype),
        shape_dtype,
        x,
    )


def _k1_jax(x):
    """K_1 for jnp arrays via scipy callback."""
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k1(xx), dtype=xx.dtype),
        shape_dtype,
        x,
    )


def yukawa_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    """2D screened Poisson (Yukawa) force.

    ``F = G q_i q_j K_1(r/λ) / (2π λ)``.  At ``r << λ`` this reduces to the
    2D Poisson kernel; at ``r >> λ`` it falls as exp(-r/λ) / sqrt(r λ).
    """
    return G * q_i * q_j * _INV_TWO_PI * _k1_jax(r_mag / lam) / lam


def yukawa_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    """2D Yukawa potential, ``V = -G q_i q_j K_0(r/λ) / (2π)``.

    Vanishes as ``r → ∞`` (unlike the 2D Poisson log potential), so total PE
    is well-defined.
    """
    return -G * q_i * q_j * _INV_TWO_PI * _k0_jax(r_mag / lam)


# ── 2D Riesz / fractional Laplacian Green's function ─────────────────────

# ``FieldSampler`` implements the operator ``-(-∇²)^α`` with Fourier symbol
# ``-|k|^(2α)`` (see ``field_sampler._build_operator_kernel``).  In d=2 the
# Green's function of this operator is the Riesz potential
#     G_α(r) = c_α · r^(2α - 2)
# with the dimensional prefactor
#     c_α = Γ(1 - α) / (2^(2α) π Γ(α)).
# The pairwise force / potential follow from F = -∇φ:
#     F_mag(r) = G · c_α · (2 - 2α) · q_i q_j / r^(3 - 2α)
#     V_pair(r) = -G · c_α · q_i q_j / r^(2 - 2α)
# At α = 1/2 the prefactor c_α = 1/(2π) and these reduce to the (2π-scaled)
# Newtonian inverse-square law that ``FieldSampler`` produces for that
# operator, so the two engines agree up to integrator-precision.


def _riesz_2d_prefactor(alpha: float) -> float:
    """``c_α = Γ(1-α) / (2^(2α) π Γ(α))`` for the 2D Riesz potential."""
    return float(
        scipy.special.gamma(1.0 - alpha)
        / (2.0 ** (2.0 * alpha) * np.pi * scipy.special.gamma(alpha))
    )


def riesz_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    """2D fractional-Laplacian (Riesz) force, ``F ∝ q_i q_j / r^(3 - 2α)``."""
    c = _riesz_2d_prefactor(alpha)
    return G * c * (2.0 - 2.0 * alpha) * q_i * q_j / r_mag ** (3.0 - 2.0 * alpha)


def riesz_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    """2D Riesz pair potential, ``V = -G c_α q_i q_j / r^(2 - 2α)``.

    Vanishes as ``r → ∞`` for α < 1.  At α = 1/2: ``c_α = 1/(2π)`` and the
    kernel matches the FieldSampler "fractional_laplacian" output.
    """
    c = _riesz_2d_prefactor(alpha)
    return -G * c * q_i * q_j / r_mag ** (2.0 - 2.0 * alpha)
