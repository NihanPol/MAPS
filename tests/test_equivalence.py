# [Claude] Numerical-equivalence regression tests for the MAPS performance work.
#
# The project rule for the perf branch is that every optimization must be
# numerically equivalent to the original implementation; this file commits
# those checks so they can be re-run after any future change (previously the
# verification lived only in throwaway /tmp scripts). In particular it
# cross-checks the TWO Clebsch-Gordan builders (pure-Python vs numba), which
# have already diverged once (the cancellation-free cg0 fix initially landed
# only in the numba kernel).
#
# Run either with pytest:
#     pytest tests/test_equivalence.py
# or standalone (no pytest needed; the package only needs to be importable):
#     python tests/test_equivalence.py
#
# Runtime is ~10-30 s (dominated by the first-call numba JIT compile, which is
# disk-cached afterwards). Tests stay at small l_max so the suite is cheap;
# they exercise the same code paths used at high l_max.

import math
import os
import sys
import tempfile
import warnings

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maps import clebschGordan as CGmod
from maps import anis_pta as ap
from maps import utils


def _skip(msg):
    print('  SKIP:', msg)


def test_cg0_closed_vs_exact():
    """Cancellation-free closed form == exact factorial Racah for all-m=0 CG."""
    worst = 0.0
    for l1 in range(0, 25):
        for l2 in range(0, 25):
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                exact = CGmod._cg_racah_exact(l1, 0, l2, 0, L, 0)
                closed = CGmod._cg0_closed(l1, l2, L)
                if (l1 + l2 + L) % 2:
                    # parity-violating cell: analytically zero. The closed form
                    # returns an exact 0; the Racah k-sum cancels only to float
                    # roundoff, so compare absolutely here.
                    assert closed == 0.0, (l1, l2, L, closed)
                    assert abs(exact) < 1e-10, (l1, l2, L, exact)
                else:
                    worst = max(worst, abs(closed - exact) / max(abs(exact), 1e-300))
    # The exact factorial evaluator is itself only ~2e-12 relative (module
    # ACCURACY note); the closed form is ~1e-13. Their disagreement is bounded
    # by the sum of the two.
    assert worst < 5e-12, worst
    print('  cg0 closed vs exact: worst rel err %.2e' % worst)


def test_cg_numba_twins_match_python():
    """The numba CG kernels must agree with their pure-Python twins.

    The formulas are maintained in two implementations (numba needs @njit);
    this is the cross-check that keeps them from silently diverging.
    """
    if not CGmod._HAVE_NUMBA:
        return _skip('numba not importable')
    lmax = 40
    lf = np.array([math.lgamma(n + 1) for n in range(4 * lmax + 4)], dtype=np.float64)
    rng = np.random.default_rng(0)
    worst0, worst1 = 0.0, 0.0
    for l1 in range(0, lmax + 1, 4):
        for l2 in range(0, lmax + 1, 4):
            for L in range(abs(l1 - l2), l1 + l2 + 1, 3):
                py = CGmod._cg0_closed(l1, l2, L)
                nb = CGmod._cg0_closed_fast(l1, l2, L, lf)
                worst0 = max(worst0, abs(py - nb))
                m1 = int(rng.integers(-l1, l1 + 1)) if l1 else 0
                m2 = int(rng.integers(-l2, l2 + 1)) if l2 else 0
                if abs(m1 + m2) <= L:
                    py1 = CGmod._cg_racah_logspace(l1, m1, l2, m2, L, m1 + m2)
                    nb1 = CGmod._cg_logspace_fast(l1, m1, l2, m2, L, m1 + m2, lf)
                    worst1 = max(worst1, abs(py1 - nb1))
    assert worst0 < 1e-13, worst0
    # The two general-m log-space implementations sum the alternating k-sum
    # differently (the Python one factors out the max term), so under the
    # k-sum's float64 cancellation they agree only to ~1e-11 absolute at
    # these l. The tolerance is set well above that roundoff floor but far
    # below any formula divergence (which shows up at O(1e-2..1)).
    assert worst1 < 1e-9, worst1
    print('  numba twins: cg0 %.2e, general-m %.2e (abs)' % (worst0, worst1))


def test_beta_python_vs_numba_builder():
    """Full beta cross-builder check: pure-Python calc_beta vs _calc_beta_fast.

    Normal dispatch only uses the numba kernel for l_max >= 64 (a ~minutes
    build), so invoke the kernel directly at l_max=12 against the Python-built
    CSR. Agreement here is what makes the builder-keyed disk caches mutually
    consistent to the documented tolerance.
    """
    if not CGmod._HAVE_NUMBA:
        return _skip('numba not importable')
    h = CGmod.clebschGordan(l_max=12)
    nfull = 2 * h.blm_size - h.blmax - 1
    _lm = np.array([h.idxtoalm(h.blmax, j) for j in range(nfull)], dtype=np.int64)
    l_arr = np.ascontiguousarray(_lm[:, 0])
    m_arr = np.ascontiguousarray(_lm[:, 1])
    almidx_arr = np.full((h.almax + 1, 2 * h.almax + 1), -1, dtype=np.int64)
    for ii in range(h.alm_size):
        L, M = h.idxtoalm(h.almax, ii)
        almidx_arr[int(L), int(M) + h.almax] = ii
    lf = np.array([math.lgamma(n + 1) for n in range(2 * h.almax + 2)], dtype=np.float64)
    rows, cols, data = CGmod._calc_beta_fast(nfull, h.almax, h.blmax,
                                             l_arr, m_arr, almidx_arr, lf, 4 * np.pi)
    beta_nb = sp.csr_matrix((data, (rows, cols)), shape=(h.alm_size, nfull * nfull))
    diff = np.abs((beta_nb - h._beta_csr).toarray()).max()
    assert diff < 1e-12, diff
    print('  beta cross-builder (l_max=12): max abs diff %.2e' % diff)


def test_blm2alm_sparse_vs_dense_einsum():
    """Sparse CSR contraction == the original dense einsum."""
    h = CGmod.clebschGordan(l_max=8)
    rng = np.random.default_rng(1)
    blms = rng.normal(size=h.blm_size) + 1j * rng.normal(size=h.blm_size)
    blms[0] = 1.0
    dense = h.beta_vals  # lazy dense reconstruction of the sparse beta
    blm_full = h.calc_blm_full(blms)
    ref = np.einsum('ijk,j,k', dense, blm_full, blm_full)
    new = h.blm_2_alm(blms)
    diff = np.abs(new - ref).max()
    assert diff < 1e-13, diff
    print('  blm_2_alm sparse vs dense einsum (l_max=8): max abs diff %.2e' % diff)


def test_cache_roundtrip_and_builder_key():
    """Disk cache: builder-keyed filename, and a reload returns the same CSR."""
    with tempfile.TemporaryDirectory() as td:
        h1 = CGmod.clebschGordan(l_max=8, cache_dir=td)
        fname = os.path.basename(h1._beta_cache_path())
        assert fname == 'maps_beta_lmax8_exact_v2.npz', fname
        assert os.path.exists(h1._beta_cache_path())
        h2 = CGmod.clebschGordan(l_max=8, cache_dir=td)  # cache hit
        assert (h1._beta_csr != h2._beta_csr).nnz == 0
        # the two builders must never share a cache key at the same l_max
        h1.almax = 64
        saved = CGmod._HAVE_NUMBA
        try:
            CGmod._HAVE_NUMBA = True
            key_nb = os.path.basename(h1._beta_cache_path())
            CGmod._HAVE_NUMBA = False
            key_py = os.path.basename(h1._beta_cache_path())
        finally:
            CGmod._HAVE_NUMBA = saved
        assert key_nb != key_py, (key_nb, key_py)
    print('  cache: roundtrip equal; keys %s / %s distinct' % (key_nb, key_py))


def test_safe_lmax_guard_raises():
    """l_max > SAFE_LMAX must refuse to build unless allow_lossy_cg=True."""
    try:
        CGmod.clebschGordan(l_max=CGmod.SAFE_LMAX + 2)
    except ValueError:
        print('  SAFE_LMAX guard: ValueError raised as expected')
        return
    raise AssertionError('clebschGordan(l_max=%d) did not raise'
                         % (CGmod.SAFE_LMAX + 2))


def _tiny_pta(l_max, nside, mode='sqrt_power_basis', **kw):
    rng = np.random.default_rng(2)
    npsr = 8
    theta = np.arccos(1 - 2 * rng.uniform(0, 1, npsr))
    phi = rng.uniform(0, 2 * np.pi, npsr)
    return ap.anis_pta(theta, phi, nside=nside, l_max=l_max, mode=mode, **kw)


def test_bandlimit_warning():
    """l_max > 3*nside-1 must warn (in any spherical-harmonic mode)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _tiny_pta(l_max=6, nside=2, mode='power_basis')
        assert any('band limit' in str(x.message) for x in w), \
            'no band-limit warning at l_max=6, nside=2'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _tiny_pta(l_max=6, nside=4, mode='power_basis')
        assert not any('band limit' in str(x.message) for x in w), \
            'spurious band-limit warning at l_max=6, nside=4'
    print('  band-limit warning: fires at (6, nside=2), silent at (6, nside=4)')


def test_signal_to_noise_forwards_cache_dir():
    """utils.signal_to_noise's internal iso_pta must inherit beta_cache_dir."""
    with tempfile.TemporaryDirectory() as td:
        pta = _tiny_pta(l_max=6, nside=8, beta_cache_dir=td)
        ncc = pta.npairs
        pta.set_data(pta.get_pure_HD(), np.repeat(0.1, ncc), 1)
        lm_out = pta.max_lkl_sqrt_power()
        utils.signal_to_noise(pta, lm_out)
        cached = sorted(os.listdir(td))
        assert 'maps_beta_lmax0_exact_v2.npz' in cached, cached
    print('  signal_to_noise: iso_pta wrote its beta cache into the forwarded dir')


def test_sqrt_loglike_finite():
    """End-to-end smoke: sqrt-basis logLikelihood is finite on synthetic data."""
    pta = _tiny_pta(l_max=6, nside=8)
    ncc = pta.npairs
    pta.set_data(pta.get_pure_HD(), np.repeat(0.1, ncc), 1)
    rng = np.random.default_rng(3)
    sample = np.concatenate([[0.0], rng.normal(0, 0.1, pta.ndim - 1)])
    ll = pta.logLikelihood(sample)
    assert np.isfinite(ll), ll
    print('  sqrt logLikelihood finite: %.6f' % ll)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        t()
    print('ALL %d TESTS PASSED' % len(tests))
