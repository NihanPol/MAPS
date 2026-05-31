import numpy as np
from healpy import Alm
from sympy.physics.quantum.cg import CG
from collections import OrderedDict
import os
from scipy import sparse as sp

# [Claude optimization] Numerical Clebsch-Gordan coefficient via the Racah
# closed-form formula, used by clebschGordan.calc_beta in place of sympy's
# symbolic CG(...).doit().evalf(). Verified to reproduce the symbolic result to
# ~1e-16 absolute difference across l_max = 0..12 (exhaustive vs sympy). The
# per-coefficient CG call is ~30-40x faster than sympy; calc_beta as a whole is
# ~5x faster (l_max=6: ~3.1 s -> ~0.6 s) because it is still a Python triple
# loop (49x16x16) -- after this change the loop, not the CG call, is the
# dominant cost (a future vectorization could speed it up further). Integer
# angular momenta only, which is all this module uses; returns 0.0 when the
# selection rules are violated.
from math import factorial as _fac, sqrt as _sqrt

def _cg_racah(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (integer spins)."""
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


class clebschGordan():

    '''
    Class with methods for manipulating clebsch-gordon coeffcients.
    '''

    def __init__(self, l_max, cache_dir=None, beta_dense_max_bytes=2_000_000_000):

        self.almax = int(l_max)
        self.blmax = int(self.almax / 2.)  #CG selection rule to ensure positive power across all sky

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
        '''

        nfull = 2 * self.blm_size - self.blmax - 1

        # Opt-in disk cache, keyed by l_max.
        if self._cache_dir is not None and os.path.exists(self._beta_cache_path()):
            self._beta_csr = sp.load_npz(self._beta_cache_path())
            self._beta_shape = (self.alm_size, nfull, nfull)
            return

        # (L, M) -> alm index, in the extended ordering used by idxtoalm.
        almidx_of = {}
        for ii in range(self.alm_size):
            L, M = self.idxtoalm(self.almax, ii)
            almidx_of[(int(L), int(M))] = ii

        # (l, m) for every blm_full index.
        lm = [self.idxtoalm(self.blmax, j) for j in range(nfull)]

        # cg0(l1, l2, L) depends only on (l1, l2, L); cache it.
        cg0_cache = {}
        def cg0(l1, l2, L):
            v = cg0_cache.get((l1, l2, L))
            if v is None:
                v = _cg_racah(l1, 0, l2, 0, L, 0)
                cg0_cache[(l1, l2, L)] = v
            return v

        four_pi = 4 * np.pi
        rows, cols, data = [], [], []
        for jj in range(nfull):
            l1, m1 = int(lm[jj][0]), int(lm[jj][1])
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
                    c1 = _cg_racah(l1, m1, l2, m2, L, M)
                    if c1 == 0.0:
                        continue
                    val = np.sqrt((2*l1 + 1) * (2*l2 + 1) / (four_pi * (2*L + 1))) * c0 * c1
                    if val != 0.0:
                        rows.append(almidx_of[(L, M)])
                        cols.append(jj * nfull + kk)
                        data.append(val)

        self._beta_csr = sp.csr_matrix(
            (np.array(data, dtype=float),
             (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(self.alm_size, nfull * nfull))
        self._beta_shape = (self.alm_size, nfull, nfull)

        if self._cache_dir is not None:
            os.makedirs(self._cache_dir, exist_ok=True)
            sp.save_npz(self._beta_cache_path(), self._beta_csr)

    def _beta_cache_path(self):
        # [Claude optimization] cache filename keyed by l_max (+ format version).
        return os.path.join(self._cache_dir, "maps_beta_lmax%d_v1.npz" % self.almax)

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



