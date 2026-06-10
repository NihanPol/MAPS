import numpy as np
from healpy import Alm
from sympy.physics.quantum.cg import CG
from collections import OrderedDict
import os
import math
import warnings
from scipy import sparse as sp

# [Claude optimization] Optional numba acceleration for the high-l_max calc_beta
# fast path (l_max >= 64). If numba is unavailable the code falls back to the
# exact Python loop (slower, but with identical low-l behaviour).
try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# [Claude optimization] Numerical Clebsch-Gordan coefficient via the Racah
# closed-form formula, used by clebschGordan.calc_beta in place of sympy's
# symbolic CG(...).doit().evalf(). The per-coefficient CG call is ~30-40x faster
# than sympy. Integer angular momenta only, which is all this module uses;
# returns 0.0 when the selection rules are violated.
#
# ACCURACY (measured 2026-05-31 vs an mpmath dps>=50 Racah oracle, restricted to
# the inputs calc_beta actually generates: l1,l2 <= blmax = l_max//2):
#   * l_max <= 63 -- exact float64 path. Worst ~2e-12 relative; unchanged
#     (bit-for-bit) from the previous committed evaluator. (NB: some stretched
#     cells already use the log-space fallback below for l >~ 32, accurate to
#     ~1e-12 -- so this is NOT bit-identical to arbitrary-precision sympy, only
#     to the committed float64 evaluator.)
#   * l_max 64..120 -- numba log-space path (_calc_beta_fast), or the Python
#     loop when numba is unavailable. In BOTH builders every all-m=0 CG (both
#     the cg0 factor AND the m1=m2=0 cg1 factor) uses a CANCELLATION-FREE
#     single-term closed form (_cg0_closed_fast in the numba kernel; the Python
#     cg0 helper falls back to _cg0_closed when the exact factorial form
#     overflows; ~1e-13 at any l), so all-m=0 beta entries are near-exact. Only
#     the cg1 factor with m1 or m2 nonzero uses the alternating Racah k-sum,
#     whose float64 catastrophic cancellation grows with l_max: worst
#     meaningful-magnitude relative error ~1e-8 (l_max=64), ~1e-6 (100),
#     ~1.7e-2 / ~2e-4 absolute (120, worst cell; typical ~1e-4).
#   * l_max > SAFE_LMAX (=120) -- the cg1 k-sum loses too many bits (>~10% error
#     by l_max~128, O(1)/sign-flipped by ~140), so clebschGordan REFUSES to build
#     by default. Pass allow_lossy_cg=True to override (with a warning) if you
#     knowingly accept the reduced accuracy.
#
# WARNING: do NOT reuse _cg_racah / _cg_racah_logspace / _cg_logspace_fast as a
# general-purpose CG routine at high l -- for inputs with all three momenta near
# l_max (which calc_beta never generates) they can lose >1e-4 absolute accuracy
# to the same cancellation.
from math import factorial as _fac, sqrt as _sqrt, lgamma as _lgamma, exp as _exp, log as _log

# [Claude] Maximum l_max for which the sqrt-power-basis Clebsch-Gordan build is
# trusted. Beyond this the general-m cg1 Racah k-sum suffers catastrophic float64
# cancellation (see ACCURACY note above). clebschGordan raises for l_max > this
# unless allow_lossy_cg=True. Calibrated against an mpmath oracle: worst
# meaningful-magnitude cg1 error <=~1.7e-2 rel / 2e-4 abs at l_max=120, rising
# steeply (>~10% by ~128, O(1) by ~140).
SAFE_LMAX = 120


def _cg_racah_logspace(j1, m1, j2, m2, J, M):
    """Overflow-safe log-space Clebsch-Gordan <j1 m1 j2 m2 | J M> (integer spins).

    Used only when the exact direct factorial form would overflow float64. Works
    in log space (math.lgamma) and factors out the largest term of the alternating
    k-sum for scale only -- this does NOT cure catastrophic cancellation. Accuracy
    is good for calc_beta's inputs up to l_max~120 (worst ~2e-4 absolute) but
    degrades sharply beyond (O(1)/sign-flipped by l_max~140); the SAFE_LMAX guard
    in clebschGordan bounds the regime. Selection rules already checked by caller.
    """
    def _lf(n):
        return _lgamma(n + 1)
    log_pref = 0.5 * (_log(2 * J + 1)
                      + _lf(j1 + j2 - J) + _lf(j1 - j2 + J) + _lf(-j1 + j2 + J) - _lf(j1 + j2 + J + 1)
                      + _lf(J + M) + _lf(J - M) + _lf(j1 - m1) + _lf(j1 + m1) + _lf(j2 - m2) + _lf(j2 + m2))
    kmin = max(0, j2 - J - m1, j1 - J + m2)
    kmax = min(j1 + j2 - J, j1 - m1, j2 + m2)
    log_terms = []
    for k in range(kmin, kmax + 1):
        log_denom = (_lf(k) + _lf(j1 + j2 - J - k) + _lf(j1 - m1 - k)
                     + _lf(j2 + m2 - k) + _lf(J - j2 + m1 + k) + _lf(J - j1 - m2 + k))
        log_terms.append(log_pref - log_denom)
    lmx = max(log_terms)
    ksum = 0.0
    for k, lt in zip(range(kmin, kmax + 1), log_terms):
        ksum += ((-1) ** k) * _exp(lt - lmx)
    return ksum * _exp(lmx)


def _cg_racah_exact(j1, m1, j2, m2, J, M):
    """Exact-factorial Clebsch-Gordan <j1 m1 j2 m2 | J M> (integer spins).

    [Claude fix] Split out of _cg_racah so callers can pick their own overflow
    fallback: raises OverflowError when the factorial products exceed float64
    (high l). _cg_racah falls back to the log-space k-sum (general m); the
    calc_beta cg0 helper falls back to the cancellation-free _cg0_closed.
    """
    # [Claude optimization] Cast to Python int up front. calc_beta passes
    # numpy.int64 indices (from healpy Alm.getlm via idxtoalm); numpy integer
    # arithmetic on the factorial products below silently overflows int64 and
    # wraps negative -> math.sqrt domain error for l_max >= 14. Python ints are
    # arbitrary precision, matching sympy's behavior on the original code path.
    j1, m1, j2, m2, J, M = int(j1), int(m1), int(j2), int(m2), int(J), int(M)
    if M != m1 + m2:
        return 0.0
    if (J < abs(j1 - j2)) or (J > j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    pref = _sqrt((2 * J + 1)
                 * _fac(j1 + j2 - J) * _fac(j1 - j2 + J) * _fac(-j1 + j2 + J)
                 / _fac(j1 + j2 + J + 1))
    pref *= _sqrt(_fac(J + M) * _fac(J - M) * _fac(j1 - m1) * _fac(j1 + m1)
                  * _fac(j2 - m2) * _fac(j2 + m2))
    ksum = 0.0
    kmin = max(0, j2 - J - m1, j1 - J + m2)
    kmax = min(j1 + j2 - J, j1 - m1, j2 + m2)
    for k in range(kmin, kmax + 1):
        ksum += ((-1) ** k) / (
            _fac(k) * _fac(j1 + j2 - J - k) * _fac(j1 - m1 - k)
            * _fac(j2 + m2 - k) * _fac(J - j2 + m1 + k) * _fac(J - j1 - m2 + k))
    return pref * ksum


def _cg_racah(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (integer spins)."""
    # [Claude optimization] Exact direct factorial form. For very high l the
    # second sqrt's factorial product exceeds float64's max -> OverflowError; in
    # that case only, fall back to the overflow-safe log-space evaluation. The
    # try has zero cost when no overflow occurs, so low-l results are unchanged.
    try:
        return _cg_racah_exact(j1, m1, j2, m2, J, M)
    except OverflowError:
        return _cg_racah_logspace(j1, m1, j2, m2, J, M)


def _cg0_closed(l1, l2, L):
    """Cancellation-free Clebsch-Gordan <l1 0 l2 0 | L 0> (integer l).

    [Claude optimization] The all-m=0 reduced coefficient has a SINGLE-TERM closed
    form (via the Wigner 3j with all m=0), so unlike the general Racah k-sum it has
    NO alternating sum and therefore NO catastrophic cancellation. Evaluated in
    log space it is accurate to ~1e-13 at any l_max. Returns 0.0 when the parity or
    triangle selection rule is violated.

        3j(l1 l2 L;0 0 0) = (-1)^g sqrt[(2g-2l1)!(2g-2l2)!(2g-2L)!/(2g+1)!]
                            * g! / [(g-l1)!(g-l2)!(g-L)!],   g = (l1+l2+L)/2
        <l1 0 l2 0|L 0>    = (-1)^(l1-l2) sqrt(2L+1) * 3j(l1 l2 L;0 0 0)
    """
    l1, l2, L = int(l1), int(l2), int(L)
    if (l1 + l2 + L) % 2:
        return 0.0
    if L < abs(l1 - l2) or L > l1 + l2:
        return 0.0
    g2 = l1 + l2 + L
    g = g2 // 2
    log_mag = (0.5 * (_lgamma(g2 - 2*l1 + 1) + _lgamma(g2 - 2*l2 + 1)
                      + _lgamma(g2 - 2*L + 1) - _lgamma(g2 + 2))
               + _lgamma(g + 1) - _lgamma(g - l1 + 1) - _lgamma(g - l2 + 1) - _lgamma(g - L + 1))
    threej = ((-1) ** g) * _exp(log_mag)
    return ((-1) ** (l1 - l2)) * _sqrt(2 * L + 1) * threej


if _HAVE_NUMBA:

    @_njit(cache=True, inline='always')
    def _cg_logspace_fast(j1, m1, j2, m2, J, M, lf):
        # [Claude optimization] numba log-space Clebsch-Gordan using a precomputed
        # log-factorial table lf[n] = log(n!). Used on the l_max>=64 fast path for
        # the GENERAL-m cg1 only (the all-m=0 cg0 uses the cancellation-free
        # _cg0_closed_fast below). The alternating k-sum suffers catastrophic
        # float64 cancellation that grows with l: accuracy is ~1e-8 at l_max=64,
        # ~2e-4 absolute at l_max=120, and O(1)/sign-flipped by ~140 -- so it is
        # only used within the SAFE_LMAX (=120) regime that clebschGordan enforces.
        if M != m1 + m2:
            return 0.0
        if J < abs(j1 - j2) or J > j1 + j2:
            return 0.0
        if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
            return 0.0
        lp = 0.5 * (math.log(2.0 * J + 1.0)
                    + lf[j1 + j2 - J] + lf[j1 - j2 + J] + lf[-j1 + j2 + J] - lf[j1 + j2 + J + 1]
                    + lf[J + M] + lf[J - M] + lf[j1 - m1] + lf[j1 + m1] + lf[j2 - m2] + lf[j2 + m2])
        kmin = max(0, j2 - J - m1, j1 - J + m2)
        kmax = min(j1 + j2 - J, j1 - m1, j2 + m2)
        s = 0.0
        for k in range(kmin, kmax + 1):
            ld = (lf[k] + lf[j1 + j2 - J - k] + lf[j1 - m1 - k]
                  + lf[j2 + m2 - k] + lf[J - j2 + m1 + k] + lf[J - j1 - m2 + k])
            sk = 1.0 if (k % 2 == 0) else -1.0
            s += sk * math.exp(lp - ld)
        return s

    @_njit(cache=True, inline='always')
    def _cg0_closed_fast(l1, l2, L, lf):
        # [Claude optimization] numba twin of _cg0_closed: cancellation-free
        # single-term <l1 0 l2 0|L 0> via the all-m=0 Wigner 3j. ~1e-13 at any l.
        if (l1 + l2 + L) % 2:
            return 0.0
        if L < abs(l1 - l2) or L > l1 + l2:
            return 0.0
        g2 = l1 + l2 + L
        g = g2 // 2
        log_mag = (0.5 * (lf[g2 - 2*l1] + lf[g2 - 2*l2] + lf[g2 - 2*L] - lf[g2 + 1])
                   + lf[g] - lf[g - l1] - lf[g - l2] - lf[g - L])
        sgn_g = 1.0 if (g % 2 == 0) else -1.0
        sgn_ll = 1.0 if ((l1 - l2) % 2 == 0) else -1.0
        return sgn_ll * math.sqrt(2.0 * L + 1.0) * sgn_g * math.exp(log_mag)

    @_njit(cache=True)
    def _calc_beta_fast(nfull, almax, blmax, l_arr, m_arr, almidx_arr, lf, four_pi):
        # [Claude optimization] numba sparse-beta builder for high l_max. cg0 is
        # tabulated once; two passes (count, then fill) produce COO arrays
        # (int32 indices, float64 data). Same selection rules and value formula
        # as the exact path, just with the log-space CG.
        cg0 = np.zeros((blmax + 1, blmax + 1, almax + 1))
        for l1 in range(blmax + 1):
            for l2 in range(blmax + 1):
                lhi = l1 + l2
                if lhi > almax:
                    lhi = almax
                for L in range(abs(l1 - l2), lhi + 1):
                    if (l1 + l2 + L) % 2 == 0:
                        # [Claude] cancellation-free closed form for the all-m=0 cg0
                        cg0[l1, l2, L] = _cg0_closed_fast(l1, l2, L, lf)
        cnt = 0
        for jj in range(nfull):
            l1 = l_arr[jj]; m1 = m_arr[jj]
            for kk in range(nfull):
                l2 = l_arr[kk]; m2 = m_arr[kk]; M = m1 + m2
                for L in range(abs(l1 - l2), l1 + l2 + 1):
                    if (l1 + l2 + L) % 2:
                        continue
                    if abs(M) > L:
                        continue
                    cnt += 1
        rows = np.empty(cnt, np.int32)
        cols = np.empty(cnt, np.int32)
        data = np.empty(cnt, np.float64)
        idx = 0
        for jj in range(nfull):
            l1 = l_arr[jj]; m1 = m_arr[jj]
            for kk in range(nfull):
                l2 = l_arr[kk]; m2 = m_arr[kk]; M = m1 + m2
                for L in range(abs(l1 - l2), l1 + l2 + 1):
                    if (l1 + l2 + L) % 2:
                        continue
                    if abs(M) > L:
                        continue
                    c0 = cg0[l1, l2, L]
                    if c0 == 0.0:
                        continue
                    # [Claude] For the all-m=0 case, c1 == c0 = <l1 0 l2 0|L 0>, so
                    # reuse the cancellation-free closed form (cg0) instead of the
                    # lossy Racah k-sum -- these are the worst cancellation cells at
                    # high l (e.g. (60,0,60,0,66,0) at l_max=120 was ~5e-3 wrong).
                    if m1 == 0 and m2 == 0:
                        c1 = c0
                    else:
                        c1 = _cg_logspace_fast(l1, m1, l2, m2, L, M, lf)
                    if c1 == 0.0:
                        continue
                    val = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (four_pi * (2 * L + 1))) * c0 * c1
                    if val != 0.0:
                        rows[idx] = almidx_arr[L, M + almax]
                        cols[idx] = jj * nfull + kk
                        data[idx] = val
                        idx += 1
        return rows[:idx], cols[:idx], data[:idx]


class clebschGordan():

    '''
    Class with methods for manipulating clebsch-gordon coeffcients.

    Warning:
        [Claude] Accuracy degrades at high l_max. The general-m Clebsch-Gordan
        coefficients used to convert the sqrt-power b_lm -> c_lm are evaluated in
        float64 via an alternating Racah sum that suffers catastrophic
        cancellation growing with l_max. Results are effectively exact for
        l_max <= 63, degrade through l_max = 64..120 (the worst
        meaningful-magnitude coefficients reach tens of percent relative error
        by l_max ~ 120, and small coefficients can even flip sign), and are
        refused above SAFE_LMAX (= 120): constructing with l_max > 120 raises
        ValueError unless allow_lossy_cg=True. Prefer l_max <= 63 for
        high-fidelity power maps / angular power spectra. (The all-m = 0
        coefficients use a cancellation-free closed form and are unaffected.)
    '''

    def __init__(self, l_max, cache_dir=None, beta_dense_max_bytes=2_000_000_000,
                 allow_lossy_cg=False):
        """Build the Clebsch-Gordan helper for the sqrt-power basis at l_max.

        [Claude] Args:
            l_max (int): Maximum multipole of the (clm) power map; the sqrt
                parameters run to blmax = l_max // 2.
            cache_dir (str, optional): If given, the sparse beta matrix is cached
                to / loaded from this directory (keyed by l_max, the builder
                path -- numba vs exact -- and a values version). Default None.
            beta_dense_max_bytes (int): Byte budget above which the dense
                ``beta_vals`` property refuses to materialize (default ~2 GB).
            allow_lossy_cg (bool): Permit construction above SAFE_LMAX despite the
                loss of accuracy described in the warning below. Default False.

        Warning:
            [Claude] Accuracy degrades at high l_max. The general-m Clebsch-Gordan
            coefficients are computed by a float64 alternating Racah sum subject
            to catastrophic cancellation that grows with l_max: effectively exact
            for l_max <= 63, degrading through l_max = 64..120 (worst meaningful
            coefficients reach tens of percent relative error by l_max ~ 120),
            and untrustworthy beyond. l_max > SAFE_LMAX (= 120) raises ValueError
            unless allow_lossy_cg=True (which only warns and proceeds with the
            reduced accuracy).
        """

        self.almax = int(l_max)
        self.blmax = int(self.almax / 2.)  #CG selection rule to ensure positive power across all sky

        # [Claude] Cancellation guard. For l_max > SAFE_LMAX the general-m cg1
        # Clebsch-Gordan k-sum loses too many bits to catastrophic float64
        # cancellation (see the ACCURACY note at the top of this module), so the
        # resulting beta -> sky map would be silently wrong. Refuse by default;
        # allow_lossy_cg=True overrides with a warning for callers who knowingly
        # accept the reduced accuracy.
        if self.almax > SAFE_LMAX:
            msg = ("clebschGordan(l_max=%d) exceeds SAFE_LMAX=%d: the general-m "
                   "Clebsch-Gordan coefficients are computed by a Racah k-sum whose "
                   "float64 catastrophic cancellation makes them untrustworthy at "
                   "this l_max (worst meaningful-cell error ~%s by l_max~128, "
                   "O(1)/sign-flipped by ~140). " % (self.almax, SAFE_LMAX, "10%"))
            if not allow_lossy_cg:
                raise ValueError(
                    msg + "Pass allow_lossy_cg=True to build anyway and accept the "
                    "reduced accuracy, or use a lower l_max.")
            warnings.warn(
                msg + "Proceeding because allow_lossy_cg=True; the recovered power "
                "map / likelihood may be inaccurate at high l.", stacklevel=2)

        ## size of arrays: for blms its only non-negative m values but for alms it is all of them
        self.alm_size = (self.almax + 1)**2
        self.blm_size = Alm.getsize(self.blmax)

        # [Claude optimization] sparse-beta config: opt-in on-disk cache (keyed by
        # l_max) and the byte budget above which the dense beta_vals property
        # refuses to materialize (default ~2 GB).
        self._cache_dir = cache_dir
        self._beta_dense_max_bytes = int(beta_dense_max_bytes)

        ## calculate and store beta
        self.calc_beta()

        ## calculate and store the output of the idxtoalm method for blmax.
        ## This will be used many times for the spherical harmonic likelihood

        ## Array of blm values for both +ve and -ve indices
        self.bl_idx = np.zeros(2*self.blm_size - self.blmax - 1, dtype='int')
        self.bm_idx = np.zeros(2*self.blm_size - self.blmax - 1, dtype='int')


        for ii in range(self.bl_idx.size):

            #lval, mval = Alm.getlm(blmax, jj)
            self.bl_idx[ii], self.bm_idx[ii] = self.idxtoalm(self.blmax, ii)

        # [Claude optimization] Precompute the constant gather/sign maps used by
        # calc_blm_full. These depend only on blmax (fixed at construction), so
        # building them once turns the per-evaluation Python loop over
        # Alm.getidx (run on every MCMC / lmfit residual in the sqrt basis) into
        # a single vectorized gather. Numerically identical to the original loop
        # (same source indices, same (-1)**|m| signs, same conjugation).
        _n_full = self.bl_idx.size
        self._blmfull_src = np.zeros(_n_full, dtype='int')   # |m| entry to read
        self._blmfull_neg = np.zeros(_n_full, dtype=bool)    # True where m < 0
        self._blmfull_sign = np.ones(_n_full, dtype='float')  # (-1)**|m| where m<0
        for jj in range(_n_full):
            _lval, _mval = self.bl_idx[jj], self.bm_idx[jj]
            self._blmfull_src[jj] = Alm.getidx(self.blmax, _lval, abs(_mval))
            if _mval < 0:
                self._blmfull_neg[jj] = True
                self._blmfull_sign[jj] = (-1) ** abs(_mval)


    def idxtoalm(self, lmax, ii):

        '''
        index --> (l, m) function which works for negetive indices too
        '''

        alm_size = Alm.getsize(lmax)

        if ii >= (2*alm_size - lmax - 1):
            raise ValueError('Index larger than acceptable')
        elif ii < alm_size:
            l, m = Alm.getlm(lmax, ii)
        else:
            l, m = Alm.getlm(lmax, ii - alm_size + lmax + 1)

            if m ==0:
                raise ValueError('Something wrong with ind -> (l, m) conversion')
            else:
                m = -m

        return l, m


    def calc_beta(self):

        '''
        Method to calculate beta array to convert from blm to alm.

        [Claude optimization] beta is exactly sparse: the Clebsch-Gordan
        selection rules (M = m1+m2, triangle |l1-l2|<=L<=l1+l2, parity
        (l1+l2+L) even, |M|<=L) force the vast majority of cells to zero. We
        iterate only over the nonzero (jj, kk, L) combinations and store the
        result as a scipy.sparse CSR matrix of shape (alm_size, nfull**2). The
        dense tensor is available on demand via the beta_vals property. Values
        are identical to the dense triple loop (same _cg_racah, same prefactor).

        Warning:
            [Claude] The general-m Clebsch-Gordan values lose accuracy to float64
            catastrophic cancellation as l_max grows (see the class docstring):
            effectively exact for l_max <= 63, tens-of-percent worst-case error
            by l_max ~ 120, and refused above SAFE_LMAX = 120.
        '''

        nfull = 2 * self.blm_size - self.blmax - 1

        # Opt-in disk cache, keyed by l_max + builder path + values version
        # (see _beta_cache_path).
        if self._cache_dir is not None and os.path.exists(self._beta_cache_path()):
            self._beta_csr = sp.load_npz(self._beta_cache_path())
            self._beta_shape = (self.alm_size, nfull, nfull)
            return

        # [Claude optimization] High-l_max fast path. For l_max >= 64 the exact
        # big-integer Racah CG is impractically slow AND overflows float64 for
        # many cells, so use a numba-compiled log-space builder (~150x faster:
        # l_max=64 303s->2s, l_max=70 516s->3s). cg0 is the cancellation-free
        # closed form (~1e-13); the general-m cg1 is the log-space Racah k-sum,
        # accurate to ~1e-8 (l_max=64) .. ~2e-4 absolute (l_max=120) and guarded
        # above SAFE_LMAX by __init__. l_max <= 63 keeps the exact path below, so
        # every existing verified result there is unchanged (bit-for-bit).
        if _HAVE_NUMBA and self.almax >= 64:
            # [Claude] The numba builder stores COO column indices (jj*nfull+kk,
            # max nfull**2-1) as int32; mirror the non-numba path's int64 guard so
            # we never silently overflow. nfull**2 < 2**31 holds for l_max < ~430,
            # far above SAFE_LMAX, so this only matters with allow_lossy_cg=True.
            if nfull * nfull >= 2 ** 31:
                raise ValueError(
                    "calc_beta numba path stores int32 column indices and would "
                    "overflow at l_max=%d (nfull**2=%d >= 2**31). Such an l_max is "
                    "computationally infeasible here anyway." % (self.almax, nfull * nfull))
            _lm = np.array([self.idxtoalm(self.blmax, j) for j in range(nfull)], dtype=np.int64)
            l_arr = np.ascontiguousarray(_lm[:, 0])
            m_arr = np.ascontiguousarray(_lm[:, 1])
            almidx_arr = np.full((self.almax + 1, 2 * self.almax + 1), -1, dtype=np.int64)
            for ii in range(self.alm_size):
                L, M = self.idxtoalm(self.almax, ii)
                almidx_arr[int(L), int(M) + self.almax] = ii
            lf = np.array([_lgamma(n + 1) for n in range(2 * self.almax + 2)], dtype=np.float64)
            rows, cols, data = _calc_beta_fast(nfull, self.almax, self.blmax,
                                               l_arr, m_arr, almidx_arr, lf, 4 * np.pi)
            self._beta_csr = sp.csr_matrix((data, (rows, cols)),
                                           shape=(self.alm_size, nfull * nfull))
            self._beta_shape = (self.alm_size, nfull, nfull)
            if self._cache_dir is not None:
                os.makedirs(self._cache_dir, exist_ok=True)
                sp.save_npz(self._beta_cache_path(), self._beta_csr)
            return

        # (L, M) -> alm index, in the extended ordering used by idxtoalm.
        almidx_of = {}
        for ii in range(self.alm_size):
            L, M = self.idxtoalm(self.almax, ii)
            almidx_of[(int(L), int(M))] = ii

        # (l, m) for every blm_full index.
        lm = [self.idxtoalm(self.blmax, j) for j in range(nfull)]

        # cg0(l1, l2, L) depends only on (l1, l2, L); cache it.
        # [Claude fix] Accuracy parity with the numba fast path: keep the exact
        # factorial form wherever it does not overflow (bit-for-bit unchanged at
        # l_max <= 63), but when it overflows at high l fall back to the
        # CANCELLATION-FREE closed form _cg0_closed -- not the lossy log-space
        # Racah k-sum. Previously, running without numba at l_max >= 64 sent
        # these worst-cancellation all-m=0 cells through the lossy k-sum
        # (~1e-8 error at l_max=64, ~5e-3 by 120), so results silently depended
        # on whether numba was importable.
        cg0_cache = {}
        def cg0(l1, l2, L):
            v = cg0_cache.get((l1, l2, L))
            if v is None:
                try:
                    v = _cg_racah_exact(l1, 0, l2, 0, L, 0)
                except OverflowError:
                    v = _cg0_closed(l1, l2, L)
                cg0_cache[(l1, l2, L)] = v
            return v

        four_pi = 4 * np.pi
        # [Claude optimization] Memory-frugal accumulation: collect each jj's
        # nonzeros into small per-row Python lists, convert to numpy arrays per
        # block, and concatenate once at the end. This keeps the bulk in
        # int32/float64 arrays (~16 bytes/nonzero) rather than one giant Python
        # list of boxed ints/floats (~3x larger), which OOMs at very high l_max.
        # int32 column indices are safe while nfull**2 < 2**31 (true to l_max
        # ~460); otherwise fall back to int64.
        idx_dtype = np.int32 if (nfull * nfull) < 2 ** 31 else np.int64
        row_blocks, col_blocks, data_blocks = [], [], []
        for jj in range(nfull):
            l1, m1 = int(lm[jj][0]), int(lm[jj][1])
            r, c, d = [], [], []
            for kk in range(nfull):
                l2, m2 = int(lm[kk][0]), int(lm[kk][1])
                M = m1 + m2
                for L in range(abs(l1 - l2), l1 + l2 + 1):
                    if (l1 + l2 + L) % 2:      # cg0 parity selection rule
                        continue
                    if abs(M) > L:             # cg1 requires |M| <= L
                        continue
                    c0 = cg0(l1, l2, L)
                    if c0 == 0.0:
                        continue
                    # [Claude fix] For all-m=0 cells c1 == c0 = <l1 0 l2 0|L 0>;
                    # reuse it. Identical value to the previous _cg_racah call at
                    # low l (same function, same cache), and at high l it carries
                    # the cancellation-free fallback -- mirroring the numba kernel.
                    if m1 == 0 and m2 == 0:
                        c1 = c0
                    else:
                        c1 = _cg_racah(l1, m1, l2, m2, L, M)
                    if c1 == 0.0:
                        continue
                    val = np.sqrt((2*l1 + 1) * (2*l2 + 1) / (four_pi * (2*L + 1))) * c0 * c1
                    if val != 0.0:
                        r.append(almidx_of[(L, M)])
                        c.append(jj * nfull + kk)
                        d.append(val)
            if d:
                row_blocks.append(np.array(r, dtype=idx_dtype))
                col_blocks.append(np.array(c, dtype=idx_dtype))
                data_blocks.append(np.array(d, dtype=float))

        if data_blocks:
            rows = np.concatenate(row_blocks)
            cols = np.concatenate(col_blocks)
            data = np.concatenate(data_blocks)
        else:
            rows = np.empty(0, idx_dtype)
            cols = np.empty(0, idx_dtype)
            data = np.empty(0, dtype=float)

        self._beta_csr = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.alm_size, nfull * nfull))
        self._beta_shape = (self.alm_size, nfull, nfull)

        if self._cache_dir is not None:
            os.makedirs(self._cache_dir, exist_ok=True)
            sp.save_npz(self._beta_cache_path(), self._beta_csr)

    def _beta_cache_path(self):
        # [Claude fix] cache filename keyed by l_max AND the builder that
        # produced the values ('numba' log-space fast path vs 'exact' Python
        # loop), plus a values version. The two builders give (slightly)
        # different floats at the same l_max >= 64, so the previous
        # l_max-only key could silently serve one builder's values to the
        # other -- e.g. a lossy high-l cache being reused after numba is
        # (un)installed. v2 also marks the cancellation-free cg0 fallback in
        # the exact path; stale *_v1.npz files are ignored and recomputed.
        builder = 'numba' if (_HAVE_NUMBA and self.almax >= 64) else 'exact'
        return os.path.join(self._cache_dir,
                            "maps_beta_lmax%d_%s_v2.npz" % (self.almax, builder))

    @property
    def beta_vals(self):
        # [Claude optimization] Lazy dense reconstruction of the sparse beta for
        # backward-compatibility/inspection. Refuses to allocate beyond the
        # configured byte budget so high l_max never silently tries ~49 GB.
        alm_size, nfull, _ = self._beta_shape
        nbytes = alm_size * nfull * nfull * 8
        if nbytes > self._beta_dense_max_bytes:
            raise MemoryError(
                "Dense beta_vals would need %.1f GB (l_max=%d); it is stored "
                "sparsely as self._beta_csr. Raise self._beta_dense_max_bytes "
                "to force dense reconstruction." % (nbytes / 1e9, self.almax))
        return np.asarray(self._beta_csr.todense()).reshape(self._beta_shape)

    def calc_blm_full(self, blms_in):

        '''
        Convert samples in blm space to blm complex values including negetive m vals

        Input:  blms ordered dictionary
        Ouput:  blms_full, list including blms with negative m vals

        '''

        # [Claude optimization] Vectorized gather using the maps precomputed in
        # __init__ (_blmfull_src / _blmfull_neg / _blmfull_sign), replacing the
        # original per-call Python loop over Alm.getidx. Numerically identical:
        # positive-m entries are copied straight from blms_in; negative-m entries
        # become (-1)**|m| * conj(blms_in[|m| entry]).
        _vals = blms_in[self._blmfull_src]
        blms_full = np.where(self._blmfull_neg,
                             self._blmfull_sign * np.conj(_vals),
                             _vals)

        return blms_full

    def blm_2_alm(self, blms_in):

        '''
        Convert complex blm values to alm complex values. This will contain both -ve m values too in the standard order
        '''

        if blms_in.size != self.blm_size:
            raise ValueError('The size of the input blm array does not match the size defined by lmax ')

        ## convert blm array into a full blm array with -m values too
        blm_full = self.calc_blm_full(blms_in)

        # [Claude optimization] Sparse contraction alm_i = sum_jk beta_ijk b_j b_k
        # = self._beta_csr @ vec(b (x) b). Avoids materializing the dense beta
        # tensor (which is ~49 GB at l_max=67).
        bb = np.outer(blm_full, blm_full).ravel()
        alm_vals = self._beta_csr.dot(bb)

        return alm_vals


    def blm_params_2_blms(self, blm_params):

        '''
        convert blm parameter values where amplitudes and phases are seperate to complex
        blm values.
        '''

        ## initialize blm_vals array
        blm_vals = np.zeros(self.blm_size, dtype='complex')

        ## this is b00, alsways set to 1
        blm_vals[0] = 1

        ## counter for blm_vals
        cnt = 0

        for lval in range(1, self.blmax + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(self.blmax, lval, mval)

                if mval == 0:
                    blm_vals[idx] = blm_params[cnt]
                    cnt = cnt + 1
                else:
                    #blm_vals[idx] = blm_params[cnt] + 1j * blm_params[cnt+1]
                    ## prior on amplitude, phase
                    blm_vals[idx] = blm_params[cnt] * np.exp(1j * blm_params[cnt+1])
                    cnt = cnt + 2

        return blm_vals



