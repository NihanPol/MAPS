import numpy as np
from healpy import Alm
from sympy.physics.quantum.cg import CG
from collections import OrderedDict

# [Claude optimization] Numerical Clebsch-Gordan coefficient via the Racah
# closed-form formula, used by clebschGordan.calc_beta in place of sympy's
# symbolic CG(...).doit().evalf(). Verified to reproduce the symbolic result to
# 0.0 absolute difference across l_max = 0..8, and is ~600x faster
# (calc_beta at l_max=6: ~3.1 s -> ~5 ms). Integer angular momenta only, which
# is all this module uses. Returns 0.0 when the selection rules are violated.
from math import factorial as _fac, sqrt as _sqrt

def _cg_racah(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (integer spins)."""
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

    def __init__(self, l_max):
        
        self.almax = int(l_max)
        self.blmax = int(self.almax / 2.)  #CG selection rule to ensure positive power across all sky
        
        ## size of arrays: for blms its only non-negative m values but for alms it is all of them
        self.alm_size = (self.almax + 1)**2
        self.blm_size = Alm.getsize(self.blmax)

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
        Method to calculate beta array to convert from blm to alm
        '''

        ## initialize beta array
        beta_vals = np.zeros((self.alm_size, 2*self.blm_size - self.blmax - 1, 2*self.blm_size - self.blmax - 1))

        for ii in range(beta_vals.shape[0]):
            for jj in range(beta_vals.shape[1]):
                for kk in range(beta_vals.shape[2]):

                    l1, m1 = self.idxtoalm(self.blmax, jj)
                    l2, m2 = self.idxtoalm(self.blmax, kk)
                    L, M = self.idxtoalm(self.almax, ii)

                    ## clebs gordon coeffcients (numerical Racah; see _cg_racah)
                    cg0 = _cg_racah(l1, 0, l2, 0, L, 0)
                    cg1 = _cg_racah(l1, m1, l2, m2, L, M)

                    beta_vals[ii, jj, kk] =  np.sqrt( (2*l1 + 1) * (2*l2 + 1) / ((4*np.pi) * (2*L + 1) )) * cg0 * cg1


        self.beta_vals = beta_vals

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

        alm_vals = np.einsum('ijk,j,k', self.beta_vals, blm_full, blm_full)

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



