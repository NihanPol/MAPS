# [Claude] End-to-end regression test mirroring docs/example_notebook_sqrt_sph_basis.ipynb.
#
# Replays the example notebook's EXACT call sequence (same seed 616, 67
# pulsars, l_max=6, nside=8, same noiseless injection) with plotting removed,
# and asserts every API call the notebook makes still works with the same
# signatures and produces finite, consistent results. The squared-SNR values
# are checked against the outputs stored in the committed notebook (loose
# tolerance: lmfit/library versions move the optimum slightly).
#
# Run with pytest, or standalone:  python tests/test_notebook_workflow.py
# Runtime ~1-2 minutes (dominated by the two lmfit fits at 2211 pulsar pairs).

import os
import sys

import numpy as np
import numpy.random as nr
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from maps import anis_pta as ap
from maps import utils
from maps import anis_coefficients as ac  # the notebook's vendored 'ac' import


def test_notebook_workflow():
    # --- configuration cells ---
    seed = 616
    rng = nr.default_rng(seed)
    lmax = 6
    nside = 8
    n_psrs = 67
    n_cc = (n_psrs * (n_psrs - 1)) // 2
    assert n_cc == 2211

    # --- pulsar positions ---
    psrs_phi = rng.uniform(low=0, high=2 * np.pi, size=n_psrs)
    psrs_theta = np.arccos(1 - 2 * rng.uniform(0, 1, size=n_psrs))

    # --- injection: isotropic + hotspot ---
    input_map = np.ones(hp.nside2npix(nside))
    vec = hp.ang2vec(270, 45, lonlat=True)
    disk_anis = hp.query_disc(nside=nside, vec=vec, radius=np.radians(10),
                              inclusive=False)
    input_map[disk_anis] += 50

    # --- construct without data (notebook signature) ---
    pta = ap.anis_pta(psrs_theta, psrs_phi, nside=nside, l_max=lmax,
                      mode='sqrt_power_basis', include_pta_monopole=False)

    # --- simulate correlations through F_mat, then set_data ---
    synth_rho = pta.F_mat @ input_map
    synth_sig = np.repeat(0.1, repeats=n_cc)
    assert np.all(np.isfinite(pta.get_pure_HD()))
    pta.set_data(synth_rho, synth_sig, 1)

    # --- max-likelihood sqrt-power fit ---
    lm_out = pta.max_lkl_sqrt_power()
    lm_params = np.array(list(lm_out.params.valuesdict().values()))
    assert lm_params.size == 17  # log10_A2 + 16 b_lm amplitude/phase params
    assert np.all(np.isfinite(lm_params))

    # --- post-processing chain ---
    lm_clm = utils.convert_blm_params_to_clm(pta, lm_params[1:])
    Cl = utils.angular_power_spectrum(lm_clm)
    orf = 10 ** lm_params[0] * pta.orf_from_clm(lm_clm, include_scale=False)
    assert lm_clm.shape == ((lmax + 1) ** 2,)
    assert Cl.shape == (lmax + 1,)
    assert orf.shape == (n_cc,)
    assert np.all(np.isfinite(lm_clm)) and np.all(np.isfinite(Cl))
    assert np.all(np.isfinite(orf))

    # --- comparison against the injected spectrum (vendored ac functions) ---
    input_clm = ac.clmFromMap_fast(input_map, lmax=lmax)
    input_Cl = utils.angular_power_spectrum(input_clm)
    assert np.all(np.isfinite(input_Cl)) and input_Cl[0] > 0

    # --- squared SNRs (eq. 17, Pol+2022) ---
    total_sn2, iso_sn2, anis_sn2 = utils.signal_to_noise(pta, lm_out)
    total_sn, iso_sn, anis_sn = np.sqrt([total_sn2, iso_sn2, anis_sn2])

    # Values printed in the committed notebook (different machine/library
    # versions): Total 96.245, Iso 86.429, Anis 42.345. Allow 5% drift.
    for name, got, ref in [('total', total_sn, 96.24486169068551),
                           ('iso', iso_sn, 86.42891399037729),
                           ('anis', anis_sn, 42.34520313215121)]:
        assert abs(got - ref) / ref < 0.05, (name, got, ref)
    print('  SNRs: total %.4f (nb 96.2449), iso %.4f (nb 86.4289), '
          'anis %.4f (nb 42.3452)' % (total_sn, iso_sn, anis_sn))

    # --- recovered sky map ---
    power = 10 ** lm_params[0] * ac.mapFromClm(lm_clm, nside=pta.nside)
    assert power.shape == (hp.nside2npix(nside),)
    assert np.all(np.isfinite(power))
    print('  notebook workflow: all calls ran with notebook signatures; '
          'outputs finite')


if __name__ == '__main__':
    test_notebook_workflow()
    print('NOTEBOOK WORKFLOW TEST PASSED')
