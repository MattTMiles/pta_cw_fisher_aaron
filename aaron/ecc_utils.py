# -*- coding: utf-8 -*-
"""
Scaffold for the eccentric-CW Fisher study (aaron_eccentric_fisher.ipynb).

Three student stubs (raise NotImplementedError until you write them):
    ecc_residual_earth   Section 2 -- eccentric Earth term
    evolve_back_peters   Sections 2-3 -- Peters (1964) evolution of (f_orb, e) + 1PN omega advance
    ecc_residual_full    Section 3 -- Earth term minus pulsar term

Everything else here is plumbing (no physics): parameter layout, SNR
rescaling, a jacfwd Fisher builder, a condition-number printer, ring
helpers that call Steve's code, and a plotting helper.

Steve's modules (utils_vectorized_fixed, utils_referee, constants) live one
directory up and are imported from there, never copied.
"""

import sys
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import utils_referee as urf
import utils_vectorized_fixed as uvf

# =============================================================================
# Eccentric parameter layout
# =============================================================================
# Sky must be indices 0, 1 (cos_theta, phi): utils_referee.compute_sky_area_schur
# treats the first two Fisher parameters as the sky block, and dOmega = 2*pi*
# sqrt(det Cov) is only an area in (cos_theta, phi).
ECC_PARAMS = (
    "cos_theta",   # 0  cos of GW-source polar angle (Steve's cw_costheta)
    "phi",         # 1  GW-source azimuth [rad]
    "cos_inc",     # 2  cos(inclination)
    "psi",         # 3  polarisation angle [rad]
    "log10_forb",  # 4  log10 AZIMUTHAL orbital frequency at tref [Hz]  (f_orb = f_GW/2 when e=0)
    "e",           # 5  eccentricity at tref
    "omega",       # 6  argument of periastron [rad]
    "l0",          # 7  mean anomaly at tref [rad]
    "log10_dL",    # 8  log10 luminosity distance [Mpc] -- the amplitude knob
    "log10_Mc",    # 9  log10 chirp mass [Msun]
)
IDX = {name: i for i, name in enumerate(ECC_PARAMS)}
N_GLOBAL = len(ECC_PARAMS)


def ecc_from_circular(base8: jax.Array, e: float = 0.0, omega: float = 0.0) -> jax.Array:
    """Map Steve's 8-vector to the 10-vector above.

    base8 = [cw_costheta, cw_phi, cw_cosinc, cw_log10_Mc, cw_log10_fgw,
             cw_log10_dist, cw_phase0, cw_psi]   (get_delay_batch order)

    Steve's orbital phase at tref is cw_phase0/2. With the convention
    orbital phase = omega + true anomaly (= omega + l at e=0), that gives
    l0 = cw_phase0/2 - omega. If your waveform uses another phase
    convention, change this mapping -- not the validation tolerance.
    """
    cos_theta, phi, cos_inc, log10_Mc, log10_fgw, log10_dist, phase0, psi = base8
    return jnp.array([
        cos_theta, phi, cos_inc, psi,
        log10_fgw - jnp.log10(2.0), e, omega, phase0 / 2.0 - omega,
        log10_dist, log10_Mc,
    ])


# =============================================================================
# STUDENT STUBS
# =============================================================================
@partial(jax.jit, static_argnames=("n_max",))
def ecc_residual_earth(params: jax.Array, toas: jax.Array, psr_pos: jax.Array,
                       n_max: int, tref: float) -> jax.Array:
    """Earth-term timing residuals of an eccentric SMBHB, all pulsars at once.

    Model: the timing-RESIDUAL harmonics (integrated strain) of Taylor et al.
    2016 (arXiv:1505.06208) Sec. 4, eqs. (21)-(22), n = 1 .. n_max, built on
    the Peters & Mathews (1963) harmonic decomposition. Do not code the strain
    h_{+,x} of Sec. 3 eqs. (10)-(11) / Barack & Cutler (2004); read those only
    to follow the derivation. Cross-check numerically against GWecc.jl.

    Evolution over the span: eqs. (21)-(22) assume F, e and omega constant
    over the data. Steve's circular model evolves the frequency across the
    10 yr; leaving that out costs 5.7e-3 of the peak residual, above the
    1e-3 validation tolerance. Periastron advance turns omega by ~0.3 rad
    over 10 yr at the fiducial source. So evaluate the harmonics
    adiabatically with (f_orb, e, omega)(t) from evolve_back_peters
    (signed dt) and the mean anomaly
        l(t) = l0 + int_tref^t (2 pi f_orb - omegadot) dt'
    (see the frequency convention below for why omegadot is subtracted).

    Frequency convention (fixed here, do not change): log10_forb is the
    AZIMUTHAL (observed) orbital frequency, 2 pi f_orb = n + omegadot, with
    n the mean motion (dl/dt). Then at e -> 0 the quadrupole sits at
    exactly f_GW = 2 f_orb, which is what the validation against Steve's
    circular model needs. Use 2 pi f_orb in the amplitude factor
    zeta * omega^(-1/3) as well. Expanding eqs. (21)-(22),
    a_k cos 2omega - b_k sin 2omega splits into sin(k l + 2 omega) and
    sin(k l - 2 omega), and c_k carries sin(k l). So harmonic k has lines at
        k f_orb - (k - 2) omegadot / 2pi   [sin(k l + 2 omega)]
        k f_orb - (k + 2) omegadot / 2pi   [sin(k l - 2 omega)]
        k f_orb -  k      omegadot / 2pi   [c_k, sin^2(iota) part]
    and only the first line of k = 2 survives at e = 0. You choose how to
    write the series; the convention is fixed. omegadot / (2 pi f_orb) =
    3.2% at the fiducial source, so a mean-motion reading of log10_forb
    shifts the quadrupole by ~3% and fails the 1e-3 check for a reason
    unrelated to the waveform.

    Phase convention: omega here is Taylor's pericenter angle gamma. At
    e -> 0 the orbital phase is l + omega, so ecc_from_circular sets
    l0 = phase0/2 - omega.

    Parameters
    ----------
    params : (10,) array in ECC_PARAMS order. Units: cos_theta, cos_inc,
        e dimensionless; phi, psi, omega, l0 [rad];
        log10_forb = log10(azimuthal f_orb / Hz); log10_dL = log10(d_L/Mpc);
        log10_Mc = log10(M_c/Msun).
    toas : (N_psr, N_toa) TOAs [s]. Subtract tref inside, as get_delay_batch does.
    psr_pos : (N_psr, 3) dimensionless unit vectors to the pulsars (PulsarBatch.pos).
        Antenna patterns: reuse uvf.create_gw_antenna_pattern_vmap.
    n_max : highest harmonic kept (int). STATIC under jit (it sets array
        shapes); changing it retraces.
    tref : reference epoch [s] at which log10_forb, e, omega, l0 are defined.

    Returns
    -------
    (N_psr, N_toa) residuals [s], same sign convention as
    uvf.get_delay_batch(..., pterm=False).

    Practical note (Bessel functions): the harmonics need J_k(n e) for
    k = n-2 .. n+2, n up to n_max, differentiable in e for the Fisher.
    Checked in this env (JAX 0.10.2) against scipy.special.jv/jvp, orders
    0..47, z = n e in [0, 33]: jax.scipy.special.bessel_jn(z, v=..., n_iter=50)
    exists and jax.grad/jacfwd run through it, but DO NOT USE IT here:
      - z = 0 (e = 0) returns NaN, values and derivatives;
      - derivatives are NaN or wrong for z <~ 0.01 (dJ0/dz at 1e-4 gave
        -5.1e5, true -5e-5), and NaN at 27 of 91 test points, so the
        e = 1e-4 checks would break;
      - n_iter=50 is only good to ~6e-8 at z ~ 30, and raising n_iter
        gives NaN at small z;
      - with v > n_iter it silently returns only n_iter+1 orders.
    Use Bessel's integral instead, evaluated with the trapezoid rule on M
    equally spaced points (spectrally accurate for this periodic integrand):
        J_k(z) = (1/2pi) int_0^{2pi} cos(k tau - z sin tau) d tau.
    With M = 128 this matched scipy to 2.5e-15 (values) and 2.0e-15
    (dJ/dz) for k <= 47, z in [0, 33], with no NaNs at z = 0, and it is
    plain jnp (cos, sin, mean), so jacfwd works. If you raise n_max or
    e_max, raise M and recheck against scipy. The power series
    sum_m (-1)^m (z/2)^(2m+k) / (m! (m+k)!) is fine for small z but loses
    ~12 digits to cancellation at z ~ 30.

    Implementation notes: vmap (or broadcast) over pulsars; keep it pure JAX
    so jax.jacfwd can differentiate through it.
    """
    raise NotImplementedError(
        "ecc_residual_earth: write the Taylor+2016 Sec. 4 eqs. (21)-(22) residual "
        "harmonic sum (n=1..n_max), evolved adiabatically with evolve_back_peters "
        "(azimuthal f_orb convention) -- read Taylor+2016 (arXiv:1505.06208) Secs. 2-4."
    )


@jax.jit
def evolve_back_peters(f_orb: jax.Array, e: jax.Array, omega: jax.Array, Mc: jax.Array,
                       dt: jax.Array, q: float = 1.0) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evolve (f_orb, e, omega) to the epoch dt seconds earlier.

    Three quantities, integrated together:
      - f_orb, e: Peters (1964) GW-driven evolution, Taylor+2016 eq. (6).
        Depends on Mc alone. Apply it to the azimuthal f_orb (the
        convention in ecc_residual_earth); the difference from applying it
        to the mean motion is beyond the order of this model.
      - omega: leading-order (1PN) periastron advance
            omegadot = 3 (2 pi f_orb)^(5/3) (G M_tot / c^3)^(2/3) / (1 - e^2),
        M_tot = Mc / eta^(3/5), eta = q / (1 + q)^2. omegadot is NOT
        constant over the lag, because f_orb and e change along it, so
        integrate it alongside f and e rather than using omegadot(t_ref) * dt.
        At the fiducial source, omegadot dt ~ 1.6 rad at xi = 10 deg and
        ~36 rad at 50 deg (L = 1 kpc).

    q is a held-fixed assumption (default 1, M_tot = 4^(3/5) Mc). It enters
    only through omegadot. Making q (or log10 M_tot) a Fisher parameter is
    Extension (a) in the notebook.

    Two routes for (f_orb, e); pick one:
      1. Closed form: F(e)/F(e0) = (sigma(e0)/sigma(e))^(3/2),
         Taylor+2016 eqs. (4)-(5) (Peters' a(e) relation). This is exact,
         but it gives F as a function of e, not of time: get e_p from dt by
         root-finding in e on t(e) = int de / (de/dt) (a 1-D quadrature).
         At e = 0, de/dt = 0, so use the circular closed form there. sigma(e)
         contains e^(12/19), whose derivative blows up at e = 0, so guard
         both the value and the gradient (the jnp.where "double-where" trick).
         omega then still needs int omegadot dt along that path.
      2. Direct ODE: integrate eq. (6) plus d omega/dt from t = 0 to t = -dt
         (fixed-step RK4 under lax.fori_loop or lax.scan is enough, and
         differentiable). Check convergence by halving the step.

    Parameters
    ----------
    f_orb : azimuthal orbital frequency at the reference epoch [Hz]
        (= f_GW/2 at e = 0).
    e : eccentricity at the reference epoch (dimensionless).
    omega : argument of periastron at the reference epoch [rad].
    Mc : chirp mass [Msun], linear, NOT log10. Convert with const.Tsun [s/Msun].
    dt : time offset [s], signed.
        Section 3: dt = L (1 - cos xi) / c >= 0 (pulsar term, back in time),
        giving f_orb_p <= f_orb, e_p >= e, omega_p <= omega.
        Section 2: dt = tref - t, so the Earth term evolves across the span
        (dt < 0 when t > tref).
    q : mass ratio m2/m1 (dimensionless), held fixed; default 1.

    Returns
    -------
    (f_orb_p [Hz], e_p [dimensionless], omega_p [rad]) at the epoch t_ref - dt.

    Must work on scalars and be jax.vmap-able over dt (one dt per pulsar,
    or per TOA).

    Testing: check (a) tests only e = 0: f_orb_p against Steve's circular
    omega_p0 (= 2 pi f_orb_p), e_p = 0, and omega_p - omega against the
    fixed-f closed form. The e > 0 branch needs your own sanity tests:
    evolve back by dt then forward by -dt and recover (f_orb, e, omega) to
    ~1e-10; e_p >= e and f_orb_p <= f_orb for dt > 0; and routes 1 and 2
    agree.
    """
    raise NotImplementedError(
        "evolve_back_peters: Peters 1964 dF/dt, de/dt (Taylor+2016 eq. 6) plus the 1PN "
        "omegadot, integrated together back by dt -- read Peters 1964, Phys. Rev. 136, B1224."
    )


@partial(jax.jit, static_argnames=("n_max", "linked"))
def ecc_residual_full(params: jax.Array, toas: jax.Array, psr_pos: jax.Array,
                      psr_dist: jax.Array, n_max: int, tref: float,
                      linked: bool = True) -> jax.Array:
    """Eccentric residuals with the pulsar term: Earth term minus pulsar term
    (Taylor+2016 eq. 19: Delta s_A(t) = s_A(t_e) - s_A(t_p)).

    Pulsar i is seen at the epoch t - dt_i, dt_i = L_i (1 - cos xi_i) / c.
    At that epoch the linked pulsar term needs:
      - (f_orb_p, e_p, omega_p) = evolve_back_peters(f_orb, e, omega, Mc, dt_i),
        with periastron advance included (q = 1 held fixed);
      - the mean anomaly over the lag,
            l_p = l0 - int (2 pi f_orb - omegadot) dt,
        with f_orb, e (and so omegadot) evolving under the integral.
        Under the azimuthal convention dl/dt = 2 pi f_orb - omegadot, so
        integrating 2 pi f_orb alone while also evolving omega counts the
        advance twice, putting the pulsar term ~1.6 rad out of phase at
        xi = 10 deg. evolve_back_peters returns only (f_p, e_p, omega_p),
        so compute this phase integral here or in a helper you add.

    Frequency convention: as in ecc_residual_earth, log10_forb is the
    AZIMUTHAL orbital frequency at tref (quadrupole at 2 f_orb when e -> 0),
    with harmonic lines k f_orb - (k -+ 2) omegadot/2pi and
    k f_orb - k omegadot/2pi. At the pulsar epoch, use f_orb_p and
    omegadot(f_orb_p, e_p) in the same way.

    linked=True  : l_p and omega_p are set by the geometry and the evolution
                   above, so there are no new parameters.
    linked=False : the pulsar-term phase is a free per-pulsar nuisance phase.
                   params carries N_psr extra entries (one phase per pulsar,
                   [rad]) after the 10 globals, as in Steve's phase_free
                   branch. The notebook never uses this branch; see Extension (b).

    Parameters
    ----------
    params : (10,) [linked] or (10 + N_psr,) [decoupled]; ECC_PARAMS first,
        same units as ecc_residual_earth (log10_forb azimuthal, angles in rad).
    toas : (N_psr, N_toa) [s].
    psr_pos : (N_psr, 3) dimensionless unit vectors.
    psr_dist : (N_psr,) pulsar distances L_i [kpc]; to light-seconds:
        L * const.kpc / const.c. It enters through dt_i, so jacfwd sees it
        when the caller builds psr_dist from Fisher params.
    n_max : static int, as in ecc_residual_earth.
    tref : reference epoch [s].
    linked : static bool switch, see above.

    Returns
    -------
    (N_psr, N_toa) residuals [s], same convention as
    uvf.get_delay_batch(..., pterm=True, phase_free=not linked).
    """
    raise NotImplementedError(
        "ecc_residual_full: Earth term minus pulsar term at t - dt_i, with "
        "(f_p, e_p, omega_p) from evolve_back_peters and l_p = l0 - int (2 pi f - omegadot) dt -- "
        "read Taylor+2016 (arXiv:1505.06208) eqs. (19)-(22) and footnote 7."
    )


# =============================================================================
# Plumbing (no physics)
# =============================================================================
def snr(residuals: jax.Array, toaerrs: jax.Array) -> jax.Array:
    """Optimal SNR, white noise: sqrt(sum (r/sigma)^2) over pulsars and TOAs."""
    return jnp.sqrt(jnp.sum((residuals / toaerrs) ** 2))


def calibrate_log10_dL(resid_fn: Callable[..., jax.Array], params: jax.Array,
                       toaerrs: jax.Array, snr_target: float,
                       idx: int = IDX["log10_dL"], args: tuple = ()) -> jax.Array:
    """Return params with the distance entry shifted so SNR == snr_target.

    Residuals scale as 1/d_L, so log10 d_L -> log10 d_L + log10(SNR/target).
    Works for any model with that scaling (idx=5 for Steve's 8-vector).
    resid_fn is called as resid_fn(params, *args).
    """
    s = snr(resid_fn(params, *args), toaerrs)
    return params.at[idx].add(jnp.log10(s / snr_target))


@partial(jax.jit, static_argnums=(0, 3))
def fisher_matrix(resid_fn: Callable[..., jax.Array], params: jax.Array,
                  toaerrs: jax.Array, free_idx: tuple[int, ...],
                  prior_diag: jax.Array | None = None, args: tuple = ()) -> jax.Array:
    """F_ab = sum (dr/dp_a)(dr/dp_b)/sigma^2 over the parameters in free_idx.

    Uses jax.jacfwd (tall Jacobian: outputs >> parameters). Parameters not
    in free_idx are held fixed. free_idx must start (0, 1) (the sky).
    prior_diag : optional (len(free_idx),) Gaussian prior precisions added
    to the diagonal (Steve: 1 per normalised pulsar distance, 0 elsewhere).
    resid_fn is called as resid_fn(params, *args); args are traced, so
    put anything that changes between calls (a PulsarBatch, sigma_L) there
    and keep resid_fn the same function object, or jit retraces every call.
    """
    assert free_idx[:2] == (0, 1), "sky (cos_theta, phi) must be Fisher params 0, 1"
    idx = jnp.array(free_idx)

    def r_free(p_free):
        return resid_fn(params.at[idx].set(p_free), *args)

    J = jax.jacfwd(r_free)(params[idx])            # (N_psr, N_toa, N_free)
    Jw = (J / toaerrs[..., None]).reshape(-1, len(free_idx))
    F = Jw.T @ Jw
    return F if prior_diag is None else F + jnp.diag(prior_diag)


def scaled_cond(F: jax.Array) -> float:
    """Condition number of D^-1/2 F D^-1/2 (D = diag F).

    This is the matrix the Schur/pinv helpers actually invert, so it is the
    number that controls round-off; raw cond(F) mostly measures units.
    A parameter the data do not constrain at all (zero column) makes the
    scaled matrix singular, so this returns inf (or a huge number), never NaN.
    """
    d = 1.0 / jnp.sqrt(jnp.maximum(jnp.diag(F), 1e-30))   # same floor as Steve's helpers
    c = float(jnp.linalg.cond(d[:, None] * F * d[None, :]))
    return c if np.isfinite(c) else float("inf")


def print_cond(label: str, F: jax.Array, flag: float = 1e12) -> float:
    """Print the scaled condition number of F; mark it if above flag or not finite."""
    c = scaled_cond(F)
    print(f"{label:<28s} scaled cond = {c:9.2e}" + ("   <-- FLAG" if not c <= flag else ""))
    return c


def ring_batch(xi_deg: float, n_psr: int, toas: jax.Array, toaerrs: jax.Array,
               L0: float, sigma_L: float, cos_theta: float, phi: float) -> uvf.PulsarBatch:
    """Steve's ring of n_psr pulsars at angular radius xi_deg about the source.
    pdist prior = [L0, sigma_L] kpc (Steve's normalised-distance convention)."""
    return uvf.pulsar_ring_generator_vmap(
        {"cw_costheta": cos_theta, "cw_phi": phi}, ang_radius=xi_deg, npsrs=n_psr,
        toas=toas, toaerrs=toaerrs, pdist=[L0, sigma_L])


@partial(jax.jit, static_argnames=("n_psr", "phase_free"))
def ring_dOmega(xi_deg, sigma_L, base8, toas, toaerrs, tref, L0, snr_target,
                n_psr: int = 20, phase_free: bool = False):
    """Section 1: Steve's ring + SNR calibration + Fisher + marginalised dOmega.

    Calls utils_referee.calibrated_fisher (the paper's SNR=10 recipe:
    distance rescaled on the linked pulsar-term SNR; unit prior on
    normalised pulsar distances; free phases unconstrained) and the paper's
    full-inverse compute_sky_area. jacfwd path (use_fwd=True), which Steve
    verified equals the production jacrev to ~1e-15. vmap over xi or sigma_L.
    """
    batch = ring_batch(xi_deg, n_psr, toas, toaerrs, L0, sigma_L, base8[0], base8[1])
    F, _, _ = urf.calibrated_fisher(batch, base8, tref, phase_free=phase_free,
                                    snr_target=snr_target, use_fwd=True)
    return urf.compute_sky_area(F, marginalized=True)


def plot_curves(x, curves: dict, xlabel: str, ylabel: str = r"$\Delta\Omega$ [deg$^2$]",
                logx: bool = False, logy: bool = True, hline: tuple[float, str] | None = None,
                title: str | None = None):
    """One line per entry of curves {label: y}. hline = (value, label) adds a
    dashed reference line. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for lab, y in curves.items():
        ax.plot(x, np.asarray(y), lw=1.8, label=lab)
    if hline is not None:
        ax.axhline(hline[0], color="k", ls="--", lw=1, label=hline[1])
    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax
