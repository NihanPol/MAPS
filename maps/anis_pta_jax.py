# This is mostly plagiarized from the rest of MAPS which was built by Kyle and Nihan
# and also anis_coefficients in ENTERPRISE which was built by Steve and Rutger
import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap
import jax.numpy as jnp
from jax.scipy.special import sph_harm_y # used for constructing spherical harmonic basis
import jaxopt # optimistix has a nice Nelder-Mead implementation
from functools import partial
import numpy as np, healpy as hp, os

def correlations2anisotropy(psrs_theta, psrs_phi, rho, S, C, basis, 
                            nside = 8, l_max = 8, maxiter = 30, tol = 1e-5):
    """
    A function to do an anisotropy search given cross-correlations and relevant data.
    
    Notes:
    This function supports batched fitting and GPU-accleration automatically. For batched fitting, supply multi-dimensional
    (up to 3 dimensions) rho, S, and C inputs corresponding to noise-marginalization, per-frequency, or both.
    For very large-scale analyses, it may be more efficient to construct your own tailored script based on the specifics of
    the analysis rather than using this function.
    The covariance matrix can be pair covariant or not; both methods are supported. This only uses the Levenberg-Marquardt 
    algorithm for now for fitting the cross-correlations. More algorithms may be added in the future.

    Args:
        psrs_theta (ArrayLike): 1d ArrayLike of theta values in radians of pulsar positions. Used in constructing the PTA response.
        psrs_phi (ArrayLike): 1d ArrayLike of phi values in radians of pulsar positions. Used in constructing the PTA response.
        rho (ArrayLike): ArrayLike, typically 1d, of cross-correlations output from Defiant. Can be supplied as a 2d array if
            noise-marginalization or per-frequency was used, and can also be supplied as a 3d array with both noise-marginalization
            and per-frequency.
        S (float or ArrayLike): float or ArrayLike containing the amplitudes output from Defiant. Should be same dimensionality as
            rho unless rho is 1d and S is scalar, which is okay as well.
        C (ArrayLike): ArrayLike corresponding to the covariance matrix output from Defiant. Should be the dimensionality of rho
            plus 1.
        basis (str): Which basis the decomposition is to be computed with. Must be 'pixel', 'radiometer', 'sqrt', or 'spherical'.
        nside (int): What Nside (resolution) to use for the pixel and radiometer bases. This is NOT ignored even if basis is neither
            pixel nor radiometer because a radiometer map is always constructed as an initial guess regardless of basis.
            Defaults to 8.
        l_max (int): What l_max (resolution parameter) to use for the sqrt and spherical bases. This is ignored if basis is pixel
            or radiometer. Defaults to 8.
        maxiter (int): How many iterations to use for solving. Defaults to 30. Ignored if basis is 'radiometer'.
        tol (float): Tolerance for ending the solve. Defaults to 1e-5. Ignored if basis is 'radiometer'.

    Raises:
        ValueError: If the number of dimensions of rho is not 1, 2, or 3, corresponding to one set of correlations or possible 
            noise-marginalization and/or per-frequency sets of correlations.
        ValueError: If basis is not one of 'pixel', 'radiometer', 'sqrt', or 'spherical'.

    Returns:
        tuple: A two-element tuple with the basis decomposition as the first output and the squared anisotropic SNR as the second.
    """
    # make sure inputs are JAX arrays
    psrs_theta = jnp.array(psrs_theta)
    psrs_phi = jnp.array(psrs_phi)
    rho = jnp.array(rho)
    C = jnp.array(C)
    
    # check for scalar amplitude
    if not hasattr(S, '__len__'):
        S = jnp.array([S])
    else:
        S = jnp.array(S)
        if S.ndim < rho.ndim:
            S = S[..., None] # [1,2,3] -> [[1],[2],[3]] (to make it compatible with vmap)
    
    # Construct responses
    pair_idx = jnp.array(jnp.triu_indices(len(psrs_theta),1)).T
    gwtheta, gwphi = hp.pix2ang(nside, np.arange(12*nside**2))
    gwtheta, gwphi = jnp.array(gwtheta), jnp.array(gwphi)
    
    R, Fp, Fc = signalResponse_fast(psrs_theta, psrs_phi, gwtheta, gwphi, pair_idx[:,0], pair_idx[:,1]) # npix x npair
    if basis in ('sqrt', 'spherical'):
        Gamma_lm = spherical_response(Fp, Fc, gwtheta, gwphi, l_max)[:, pair_idx[:,0], pair_idx[:,1]] # nclm x npair

    # check for potential vmap opportunities and jit-compile too while we're at it
    if rho.ndim == 1:
        normal_vmapper = lambda f : jit(f)
        radio_vmapper = lambda f : jit(f)
        pixel_vmapper = lambda f : jit(f)
        sph_vmapper = lambda f : jit(f)
        sqrt_vmapper = lambda f : jit(f)
        harmonics_x0_vmapper = lambda f : f # this vmapper is used for constructing initial clms / blms given radiometer maps
    elif rho.ndim == 2:
        normal_vmapper = lambda f : vmap(jit(f)) # normal vmap; vmap over all arguments
        radio_vmapper = lambda f : vmap(jit(f), (0,0,0,0,None)) # vmap over rho, N_inv, Lt, S, but not R
        pixel_vmapper = lambda f : vmap(jit(f), (0,0,0,0,None,0,None,None)) # vmap over rho, S, Lt, and initial params but nothing else
        sph_vmapper = lambda f : vmap(jit(f), (0,0,0,None,0,None,None)) # same as above
        sqrt_vmapper = lambda f : vmap(jit(f), (0,0,0,None,0,None,None,None,None)) # same as above
        harmonics_x0_vmapper = lambda f : vmap(f, (0,None)) # vmap over radiometer maps but not cached masks
    elif rho.ndim == 3:
        normal_vmapper = lambda f : vmap(vmap(jit(f)))
        radio_vmapper = lambda f : vmap(vmap(jit(f), (0,0,0,0,None)), (0,0,0,0,None))
        pixel_vmapper = lambda f : vmap(vmap(jit(f), (0,0,0,0,None,0,None,None)), (0,0,0,0,None,0,None,None))
        sph_vmapper = lambda f : vmap(vmap(jit(f), (0,0,0,None,0,None,None)), (0,0,0,None,0,None,None))
        sqrt_vmapper = lambda f : vmap(vmap(jit(f), (0,0,0,None,0,None,None,None,None)), (0,0,0,None,0,None,None,None,None))
        harmonics_x0_vmapper = lambda f : vmap(vmap(f, (0,None)), (0,None))
    else:
        raise ValueError('rho has a number of dimensions outside of [1,3] but is assumed to have dimension 1, 2, or 3 in this function')

    # Get basis decomposition
    N_inv = normal_vmapper(jnp.linalg.inv)(C)
    Lt = normal_vmapper(lambda N : jnp.linalg.cholesky(N).T)(N_inv)
    
    radiometer_output = radio_vmapper(radiometer_basis)(rho, N_inv, Lt, S, R)
    if basis == 'radiometer':
        basis_decomposition = radiometer_output
    else:
        radiometer_map = radiometer_output['pixel_map'] # use radiometer as initial params
        if basis == 'sqrt':
            blmax = l_max // 2
            lvals = jnp.concatenate([jnp.repeat(jnp.arange(blmax+1), jnp.array([2*ll + 1 for ll in range(blmax+1)]))])
            mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(blmax+1)])
            radiometer2blm_cache = (blmax, lvals, mvals, gwtheta, gwphi)
            params = harmonics_x0_vmapper(radiometer2blm_params)(radiometer_map, radiometer2blm_cache)
            
            alm2clm_cache = make_alm2clm_cache(l_max)
            blm2alm_cache = make_blm2alm_cache(l_max)
            basis_decomposition = sqrt_vmapper(sqrt_basis)(rho, S, Lt, Gamma_lm, params,
                                                      alm2clm_cache, blm2alm_cache, maxiter, tol)
        elif basis == 'pixel':
            basis_decomposition = pixel_vmapper(pixel_basis)(rho, S, N_inv, Lt, R, radiometer_map, maxiter, tol)
        elif basis == 'spherical':
            lvals = jnp.concatenate([jnp.repeat(jnp.arange(l_max+1), jnp.array([2*ll + 1 for ll in range(l_max+1)]))])
            mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(l_max+1)])
            radiometer2clm_cache = (l_max, lvals, mvals, gwtheta, gwphi)
            params = harmonics_x0_vmapper(radiometer2clm_params)(radiometer_map, radiometer2clm_cache)
            basis_decomposition = sph_vmapper(sph_basis)(rho, S, Lt, Gamma_lm, params, maxiter, tol)
        else:
            raise ValueError("basis must be one of 'radiometer', 'sqrt', 'pixel', 'spherical'")
            
    # Get squared SNR
    if basis in ('sqrt', 'spherical'):
        clms = basis_decomposition['clm']
        anis_orf = normal_vmapper(lambda clm : clm @ Gamma_lm)(clms)
    else:
        pixel_maps = basis_decomposition['pixel_map']
        anis_orf = normal_vmapper(lambda pix : R @ pix)(pixel_maps)
    A2 = basis_decomposition['A2']
    anis_orf *= A2
    HD = get_pure_HD(get_xi(psrs_theta, psrs_phi, pair_idx))
    iso_orf = normal_vmapper(lambda S : S*HD)(S)
    snr2 = normal_vmapper(orf2snr)(rho, iso_orf, anis_orf, N_inv)
    
    return basis_decomposition, snr2

def get_radec(psrs_theta, psrs_phi):
    """
    Get the pulsar positions in RA and DEC.

    Args:
        psrs_theta (jax.Array or np.ndarray): Pulsar theta coordinates in radians.
        psrs_phi (jax.Array or np.ndarray): Pulsar phi coordinates in radians.

    Returns:
        tuple: A two-element tuple with the first being pulsar positions in right ascension and the second being declination.
    """
    psr_ra = psrs_phi
    psr_dec = (jnp.pi/2) - psrs_theta
    return psr_ra, psr_dec

@jit
def get_xi(psrs_theta, psrs_phi, pair_idx):
    """Calculate the angular separation between pulsar pairs.

    A function to compute the angular separation between pulsar pairs.

    Args:
        psrs_theta (jax.Array or np.ndarray): 1d array of theta values in radians of pulsar positions.
        psrs_phi (jax.Array or np.ndarray): 1d array of phi values in radians of pulsar positions.
        pair_idx (jax.Array or np.ndarray): 2d array of indices of pulsars for each pulsar pair.
    Returns:
        jax.Array: A 1d array of pair separations in radians.
    """
    psrs_ra, psrs_dec = get_radec(psrs_theta, psrs_phi)

    x = jnp.cos(psrs_ra)*jnp.cos(psrs_dec)
    y = jnp.sin(psrs_ra)*jnp.cos(psrs_dec)
    z = jnp.sin(psrs_dec)

    pos_vectors = jnp.array([x,y,z])

    a,b = pair_idx[:,0], pair_idx[:,1]

    xi = jnp.zeros( len(a) )
    # This effectively does a dot product of pulsar position vectors for all pairs a,b
    pos_dot = jnp.einsum('ij,ij->j', pos_vectors[:,a], pos_vectors[:, b])
    xi = jnp.arccos( pos_dot )

    return jnp.squeeze(xi)

@jit
def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a, pair_idx_a, pair_idx_b):
    """
    A function to get the PTA response matrix (npair by npix). Adapted from ENTERPRISE.

    Args:
        ptheta_a (jax.Array or np.ndarray): A 1d array containing the pulsar theta coordinates in radians.
        pphi_a (jax.Array or np.ndarray): A 1d array containing the pulsar phi coordinates in radians.
        gwtheta_a (jax.Array or np.ndarray): A 1d array containing the GW theta coordinates in radians.
        gwphi_a (jax.Array or np.ndarray): A 1d array containing the GW phi coordinates in radians.
        pair_idx_a (jax.Array or np.ndarray): A 1d array of length N_pair containing the indices of pulsar a.
        pair_idx_b (jax.Array or np.ndarray): A 1d array of length N_pair containing the indices of pulsar b.

    Returns:
        tuple: A three-element tuple with the first element being the response matrix and the last two
            being the responses to plus and cross polarized graviational waves, respectively.
    """
    gwphi, pphi = jnp.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = jnp.meshgrid(gwtheta_a, ptheta_a)
    p = jnp.array([jnp.cos(pphi) * jnp.sin(ptheta), jnp.sin(pphi) * jnp.sin(ptheta), jnp.cos(ptheta)])
    Fp, Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p)
    R = Fp[pair_idx_a]*Fp[pair_idx_b] + Fc[pair_idx_a]*Fc[pair_idx_b]
    return R, Fp, Fc
    
def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p):
    """
    A function to get the plus and cross polarized response matrices. Adapted from ENTERPRISE.

    Args:
        pphi (ArrayLike): ArrayLike containing the pulsar phi coordinates.
        ptheta (ArrayLike): ArrayLike containing the pulsar theta coordinates.
        gwphi (ArrayLike): ArrayLike containing the GW phi coordinates.
        gwtheta (ArrayLike): ArrayLike containing the GW theta coordinates.
        p (jax.Array or np.ndarray): Array containing the pulsar position unit vectors.

    Returns:
        tuple: A two-element tuple with the first element being the plus-polarized response and second being the cross-polarized
            response.
    """
    Omega = jnp.array([-jnp.sin(gwtheta) * jnp.cos(gwphi), -jnp.sin(gwtheta) * jnp.sin(gwphi), -jnp.cos(gwtheta)])
    mhat = jnp.array([-jnp.sin(gwphi), jnp.cos(gwphi), jnp.zeros(gwphi.shape)])
    nhat = jnp.array([-jnp.cos(gwphi) * jnp.cos(gwtheta), -jnp.cos(gwtheta) * jnp.sin(gwphi), jnp.sin(gwtheta)])
    npixels = Omega.shape[2]
    c = jnp.sqrt(1.5) / jnp.sqrt(npixels)
    Fp = 0.5 * c * (jnp.sum(nhat * p, axis=0) ** 2 - jnp.sum(mhat * p, axis=0) ** 2) / (1 - jnp.sum(Omega * p, axis=0))
    Fc = c * jnp.sum(mhat * p, axis=0) * jnp.sum(nhat * p, axis=0) / (1 - jnp.sum(Omega * p, axis=0))
    return Fp, Fc

def spherical_response(Fp, Fc, gwtheta, gwphi, l_max):
    """A function to compute the spherical harmonic basis antenna response matrix R_{clm, a, b}.

    This function computes the spherical harmonics basis antenna response 
    matrix R_{clm, a, b} where a and b are pulsars.
    NOTE: This function uses the GW propogation direction for gwtheta and gwphi
    rather than the source direction (i.e. this method uses the vector from the
    source to the observer)

    Returns:
        jax.Array: An array of shape (nclm, npsr, npsr) containing the antenna
            pattern response matrix.
    """
    FpFc = jnp.zeros((Fp.shape[0], 2*Fp.shape[1])).at[:,0::2].set(Fp).at[:,1::2].set(Fc)

    lvals = jnp.concatenate([jnp.repeat(jnp.arange(l_max+1), jnp.array([2*ll + 1 for ll in range(l_max+1)]))])
    mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(l_max+1)])
    
    ylm_maps = compute_ylm_maps(lvals, mvals, gwtheta, gwphi, l_max)

    return _spherical_response(FpFc, ylm_maps)

@jit
def get_pure_HD(xi):
    """Calculate the Hellings and Downs correlation for each pulsar pair.

    This function calculates the Hellings and Downs correlation for each pulsar
    pair.

    Args:
        xi (ArrayLike): A 1d ArrayLike containing the angular separations of the pulsars in each pair (in radians).

    Returns:
        jax.Array: An array of HD correlation values for each pulsar pair.
    """
    #Return the theoretical HD curve given xi

    xx = (1 - jnp.cos(xi)) / 2.
    hd_curve = 1.5 * xx * jnp.log(xx) - xx / 4 + 0.5

    return hd_curve

def make_alm2clm_cache(l_max):
    """
    A function to get square-root basis masks needed as inputs to alm2clm.

    Args:
        l_max (int): The l_max parameter of the search.

    Returns:
        list: The spherical harmonic basis masks for use in alm2clm.
    """
    alm2clm_cache = []
    lvals = jnp.concatenate([jnp.repeat(jnp.arange(l_max+1), jnp.array([2*ll + 1 for ll in range(l_max+1)]))])
    mvals = jnp.concatenate([jnp.arange(-ll, ll+1) for ll in range(l_max+1)])
    abs_mvals = jnp.abs(mvals)
    clm_mask = abs_mvals * (2 * l_max + 1 - abs_mvals) // 2 + lvals
    mvals_positive = mvals > 0
    mvals_negative = mvals < 0
    mvals_float_power = jnp.float_power(-1, mvals)
    for mask in (clm_mask, mvals_float_power, mvals_positive, mvals_negative):
        alm2clm_cache.append(mask)
    return alm2clm_cache
    
def make_blm2alm_cache(l_max):
    """
    A function to get square-root basis masks needed as inputs to blm2alm.

    Args:
        l_max (int): The l_max parameter of the search (not divided by 2 yet, i.e., not blmax).

    Returns:
        list: The square-root basis masks for use in blm2alm.
    """
    blm2alm_cache = []
    blmax = l_max // 2
    precomputed_CG_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'precomputed_clebschGordan')
    if not os.path.exists(precomputed_CG_directory):
        os.makedirs(precomputed_CG_directory, exist_ok=True)
    precomputed_CG_filename = os.path.join(precomputed_CG_directory, 'lmax'+str(l_max)+'.npz')
    if os.path.exists(precomputed_CG_filename):
        sqrt_basis_helper = np.load(precomputed_CG_filename)
        beta_vals = sqrt_basis_helper['beta_vals']
        blvals = sqrt_basis_helper['blvals']
        bmvals = sqrt_basis_helper['bmvals']
    else:
        print('No precomputed CG coefficients found for this l_max. Computing them from scratch...')
        from . import clebschGordan as CG
        sqrt_basis_helper = CG.clebschGordan(l_max = l_max)
        print('Done.')
        beta_vals = jnp.array(sqrt_basis_helper.beta_vals)
        blvals = jnp.array(sqrt_basis_helper.bl_idx)
        bmvals = jnp.array(sqrt_basis_helper.bm_idx)

        print('Saving computed CG coefficients to MAPS directory for next time.')
        np.savez_compressed(precomputed_CG_filename, beta_vals=beta_vals, blvals=blvals, bmvals=bmvals)
        print('Done saving.')
        
    abs_bmvals = jnp.abs(bmvals)

    blm_mask = abs_bmvals * (2*blmax+1 - abs_bmvals) // 2 + blvals
    blm_vals_float_power = jnp.float_power(-1, bmvals)
    bmvals_negative = bmvals < 0
    bmvals_zero = bmvals[bmvals >= 0] == 0
    for mask in (blm_mask, blm_vals_float_power, bmvals_negative, beta_vals, bmvals_zero):
        blm2alm_cache.append(mask)
    return blm2alm_cache

def alm2clm(alm, alm2clm_cache):
    """
    A function to compute clm given alm. Adapted from ENTERPRISE.

    Args:
        alm (ArrayLike): A 1d array of alm values to be converted into clm values.
        sph_cache (list): A list of masks used during the conversion. Get this from make_sph_cache.

    Returns:
        ArrayLike: A 1d array of clm values.
    """
    clm = alm[alm2clm_cache[0]]
    clms_with_positive_m = alm2clm_cache[1] * jnp.real(clm) * jnp.sqrt(2)
    clms_with_negative_m = alm2clm_cache[1] * jnp.imag(clm) * jnp.sqrt(2)
    clm = jnp.where(alm2clm_cache[2], clms_with_positive_m, clm)
    clm = jnp.where(alm2clm_cache[3], clms_with_negative_m, clm)
    clm = jnp.real(clm)
    return clm

def blm2alm(blms, blm2alm_cache):
    """
    A function to compute a set of alms given blms.

    Args:
        blms (ArrayLike): A 1d array of blm values to be converted into alm values.
        sqrt_cache (list): A list of masks used during the conversion. Get this from make_sqrt_cache.

    Returns:
        ArrayLike: A 1d array of alm values.
    """
    blm_full = blms[blm2alm_cache[0]]
    blms_with_negative_m = blm2alm_cache[1]*jnp.conj(blm_full)
    blm_full = jnp.where(blm2alm_cache[2], blms_with_negative_m, blm_full)
    alm_vals = jnp.einsum('ijk,j,k', blm2alm_cache[3], blm_full, blm_full)
    return alm_vals

def radiometer_basis(rho, N_inv, Lt, S, R):
    """
    A function to get the radiometer basis decomposition.

    Args:
        rho (jax.Array or np.ndarray): A 1d array of cross-correlations.
        N_inv (jax.Array or np.ndarray): A 2d array of the inverted covariance matrix.
        R (jax.Array or np.ndarray): The PTA response matrix (npix x npair).

    Returns:
        dict: A dictionary with keys 'basis' labeling what basis was used (radiometer), 'pixel_map' with
            the normalized, recovered radiometer decomposition, and 'pixel_map_err' with the normalized
            uncertainties on the decomposition
    """
    fisher_matrix_radiometer = jnp.diag( R.T @ N_inv @ R )
    dirty_map = R.T @ N_inv @ rho
    fisher_diag_inv = jnp.diag( 1/fisher_matrix_radiometer )
    radio_map = fisher_diag_inv @ dirty_map
    norm = radio_map.shape[0] / jnp.sum(radio_map)
    radio_map_n = radio_map * norm
    radio_map_err = jnp.sqrt(jnp.diag(fisher_diag_inv)) * norm

    model_orf = R @ radio_map_n
    A2 = (model_orf.T @ N_inv @ rho) / (model_orf.T @ N_inv @ model_orf)
    
    out = {}
    out['A2'] = A2
    out['pixel_map'] = radio_map_n
    out['pixel_map_err'] = radio_map_err
    return out

def pixel_basis(rho, S, N_inv, Lt, R, radiometer_map, maxiter=500, tol=1e-5):
    """
    A function to get the pixel basis decomposition.

    Args:
        rho (jax.Array or np.ndarray): A 1d array of cross-correlations.
        S (jax.Array or np.ndarray): A 1d array with one element, which is the optimal statistic amplitude.
        Lt (jax.Array or np.ndarray): A 2d array corresponding to the cholesky decomposition of the inverted
            covariance matrix.
        R (jax.Array or np.ndarray): The PTA response matrix (npix x npair).
        maxiter (int): The maximum number of steps to run the solve for. Defaults to 30.
        tol (float): The absolute tolerance to stop the solver early. Defaults to 1e-5.
        materialize_jac (bool): Whether to materialize the Jacobian in the Levenberg-Marquardt algorithm.
            Defaults to True.

    Returns:
        dict: A dictionary with keys 'basis' labeling what basis was used (pixel), 'pixel_map' with the
            normalized, recovered pixel decomposition, 'A2' with the amplitude of the ORF (or, equivalently,
            pixel map), and 'state' with the state of the final step of the JAXopt solve.
    """
    params = jnp.where(radiometer_map < 0, jnp.min(jnp.abs(radiometer_map)), radiometer_map)
    params = jnp.concatenate((jnp.zeros(1), jnp.log10(params)))
    def residuals(params):
        pixel_map = 10**params[1:]
        model_orf = R @ pixel_map
        A2 = 10**(2*jnp.sin(params[0])+jnp.log10(S))
        r = rho - A2*model_orf
        return r.T @ N_inv @ r
    opt_params, state = jaxopt.LBFGS(residuals, jit=True, maxiter=maxiter, tol=tol).run(params)
    opt_pixel_map = 10**opt_params[1:]
    opt_model_orf = R @ opt_pixel_map
    opt_A2 = 10**(2*jnp.sin(opt_params[0])+jnp.log10(S))
    
    out = {}
    out['state'] = state
    out['A2'] = opt_A2
    out['pixel_map'] = opt_pixel_map
    return out

def sph_basis(rho, S, Lt, Gamma_lm, params, maxiter=30, tol=1e-5):
    """
    A function to get the spherical harmonic basis decomposition.

    Args:
        rho (jax.Array or np.ndarray): A 1d array of cross-correlations.
        S (jax.Array or np.ndarray): A 1d array with one element, which is the optimal statistic amplitude.
        Lt (jax.Array or np.ndarray): A 2d array corresponding to the cholesky decomposition of the inverted
            covariance matrix.
        Gamma_lm (jax.Array or np.ndarray): The PTA response matrix (nclm x npair).
        maxiter (int): The maximum number of steps to run the solve for. Defaults to 30.
        tol (float): The absolute tolerance to stop the solver early. Defaults to 1e-5.

    Returns:
        dict: A dictionary with keys 'basis' labeling what basis was used (spherical), 'clm' with the
            normalized, recovered clms, 'A2' with the amplitude of the ORF, and 'state' with the state 
            of the final step of the JAXopt solve.
    """
    c00 = jnp.array([jnp.sqrt(4*jnp.pi)])
    def residuals(params):
        A2 = 10**(2*jnp.sin(params[0]) + jnp.log10(S))
        clm = jnp.concatenate( (c00,params[1:]) )
        orf = clm @ Gamma_lm
        r = rho - A2*orf
        return Lt @ r
    opt_params, state = jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True, maxiter=maxiter, tol=tol).run(params)
    opt_A2 = 10**(2*jnp.sin(opt_params[0]) + jnp.log10(S))
    opt_clm = jnp.concatenate( (c00,opt_params[1:]) )
    
    out = {}
    out['state'] = state
    out['A2'] = opt_A2
    out['clm'] = opt_clm
    return out

def sqrt_basis(rho, S, Lt, Gamma_lm, params, alm2clm_cache, blm2alm_cache, maxiter=30, tol=1e-5):
    """
    A function to get the square-root spherical harmonic basis decomposition.

    Args:
        rho (jax.Array or np.ndarray): A 1d array of cross-correlations.
        S (jax.Array or np.ndarray): A 1d array with one element, which is the optimal statistic amplitude.
        Lt (jax.Array or np.ndarray): A 2d array corresponding to the cholesky decomposition of the inverted
            covariance matrix.
        Gamma_lm (jax.Array or np.ndarray): The PTA response matrix (nclm x npair).
        alm2clm_cache (list): Masks for use in alm2clm. Get this list from make_alm2clm_cache.
        blm2alm_cache (list): Masks for use in blm2alm. Get this list from make_blm2alm_cache.
        maxiter (int): The maximum number of steps to run the solve for. Defaults to 30.
        tol (float): The absolute tolerance to stop the solver early. Defaults to 1e-5.

    Returns:
        dict: A dictionary with keys 'basis' labeling what basis was used (sqrt), 'clm' with the
            normalized, recovered clms, 'A2' with the amplitude of the ORF, and 'state' with the state 
            of the final step of the JAXopt solve.
    """
    b00 = jnp.ones(1)
    def residuals(params):
        A2 = 10**(2*jnp.sin(params[0])+jnp.log10(S))
        blm = jnp.concatenate( (b00, params[1::2]+1j*params[2::2]) ) # After A2, the parameters alternate between real and imaginary components.
        blm = jnp.where(blm2alm_cache[4], jnp.real(blm), blm) # bl0 is real rather than complex.
        alm = blm2alm(blm, blm2alm_cache)
        clm = alm2clm(alm, alm2clm_cache)
        orf = clm @ Gamma_lm
        r = rho - A2*orf
        return Lt @ r
    opt_params, state = jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True, maxiter=maxiter, tol=tol).run(params)
    opt_A2 = 10**(2*jnp.sin(opt_params[0])+jnp.log10(S))
    opt_blm = jnp.concatenate( (b00, opt_params[1::2]+1j*opt_params[2::2]) )
    opt_blm = jnp.where(blm2alm_cache[4], jnp.real(opt_blm), opt_blm)
    opt_alm = blm2alm(opt_blm, blm2alm_cache)
    opt_clm = alm2clm(opt_alm, alm2clm_cache)
    
    out = {}
    out['state'] = state
    out['A2'] = opt_A2
    out['clm'] = opt_clm
    return out

def orf2snr(rho, iso_orf, anis_orf, N_inv):
    """
    A function to get the squared anisotropic SNR.

    Args:
        iso_orf (jax.Array or np.ndarray): A 1d array of cross-correlations given by amplitude times the HD correlations.
            Used in the null hypothesis.
        anis_orf (jax.Array or np.ndarray): A 1d array of a best-fit anisotropic ORF scaled by amplitude.
        N_inv (jax.Array or np.ndarray): A 2d array corresponding to the inverted covariance matrix.
        
    Returns:
        float: The squared anisotropic SNR.
    """
    anis_res = rho - anis_orf
    iso_res = rho - iso_orf
    snm = -(anis_res.T @ N_inv @ anis_res)
    hdnm = -(iso_res.T @ N_inv @ iso_res)
    anis_sn2 = snm - hdnm
    return anis_sn2

@partial(jit, static_argnums=(4,))
def compute_ylm_maps(n, m, theta, phi, lmax):
    """
    A function to compute the real spherical harmonics on a HEALPix grid. Computes Y_nm(theta, phi) for each pair (n,m)
    in zip(n,m) and each pair (theta,phi) in zip(theta,phi). Adapted from ENTERPRISE.

    Args:
        n (jax.Array or np.ndarray): An array of l values of the real spherical harmonics to be computed.
        m (jax.Array or np.ndarray): An array of m values of the real spherical harmonics to be computed.
        theta (jax.Array or np.ndarray): The theta values at which to evalute the real spherical harmonics.
        phi (jax.Array or np.ndarray): The phi values at which to evaluate the real spherial harmonics.
        lmax (int): Set this equal to or larger than the largest value in n. Just an argument to pass to JAX.

    Returns:
        jax.Array: The real spherical harmonics evaluated at every pair in zip(theta, phi).
            Has size len(n) by len(theta).
    """
    # _compute_ylm_maps() is used in constructing the spherical harmonic basis
    def evaluate_ylm_at_point(n, m, theta, phi, lmax):
        # evaluate_ylm_at_point() is a wrapper to allow integer n,m to be passed instead of arrays so that vmap will work smoothly
        return sph_harm_y(jnp.array([n]), jnp.array([m]), theta, phi, n_max=lmax)[0]

    evaluate_ylm_on_map = vmap(evaluate_ylm_at_point, in_axes=(None,None,0,0,None)) # vmap over theta and phi
    complex_output = vmap(evaluate_ylm_on_map, in_axes=(0,0,None,None,None))(n, m, theta, phi, lmax) # vmap over l and m
    output = jnp.where((m[:,None] > 0),
                    (complex_output + jnp.conj(complex_output)) / jnp.sqrt(2),
                     complex_output)
    output = jnp.where((m[:,None] < 0),
                 jnp.float_power(-1., m[:,None])*(jnp.conj(complex_output) - complex_output) / (1j*jnp.sqrt(2)),
                     output)
    return jnp.real(output)

@jit
def _spherical_response(FpFc, ylm_maps):
    """
    A function to compute the PTA response matrix in the spherical harmonic basis. Adapted from ENTERPRISE.

    Args:
        FpFc (jax.Array or np.ndarray): The PTA response matrices interweaved by polarization into a single matrix which has
            size npsr by 2*npix.
        ylm_maps (jax.Array or np.ndarray): The spherical harmonics evaluated on a HEALPix grid. Get this from _compute_ylm_maps.
            Should be size nclm by npix.
    Returns:
        jax.Array: The response matrix of size nclm by npsr by npsr.
    """
    ylm_maps_both_polarizations = jnp.repeat(ylm_maps, 2).reshape(ylm_maps.shape[0], -1)

    hdcov_F = jnp.dot(FpFc * ylm_maps_both_polarizations[:,None], FpFc.T)

    def add_pulsar_term(cov):
        return cov + jnp.diag(jnp.diag(cov))

    basis = vmap(add_pulsar_term)(hdcov_F)
    return basis

def radiometer2clm_params(radiometer_map, radiometer2clm_cache):
    """
    A function to get clms corresponding to a radiometer (or any pixel) map using orthonormality of the spherical harmonics.

    Notes:
        Only approximates the shape of the map and does not accurately transform amplitude.

    Args:
        radiometer_map (jax.Array or np.ndarray):
        radiometer2clm_cache (tuple): Just some useful arrays for the manipulation. Look in the code for correlations2anisotropy 
            to see what goes into this, but this function is not really intended to be used by most users anyway as the main
            API is through correlations2anisotropy.
            
    Returns:
        jax.Array: The clm coefficients.
    """
    l_max = radiometer2clm_cache[0]
    lvals = radiometer2clm_cache[1]
    mvals = radiometer2clm_cache[2]
    gwtheta, gwphi = radiometer2clm_cache[3], radiometer2clm_cache[4]

    clms = jnp.dot(compute_ylm_maps(lvals, mvals, gwtheta, gwphi, l_max), radiometer_map) * 4*jnp.pi/radiometer_map.shape[0]
    clms *= jnp.sqrt(4*jnp.pi)/clms[0] # normalize
    return jnp.concatenate( (jnp.array([0]), clms[1:]) ) # concatenate logA2 and remove c00

def radiometer2blm_params(radiometer_map, radiometer2blm_cache):
    """
    A function to get blm parameters corresponding to a radiometer (or any pixel) map.

    Notes:
        The strategy is to take the square root of the map, decompose it into clms, turn the clms into blms, 
        and finally turn the blms into blm parameters.
        Only approximates the shape of the map and does not accurately transform amplitude.

    Args:
        radiometer_map (jax.Array or np.ndarray): The map to be turned into blm parameters.
        radiometer2blm_cache (tuple): Just some useful arrays for the manipulation. Look in the code for correlations2anisotropy 
            to see what goes into this, but this function is not really intended to be used by most users anyway as the main
            API is through correlations2anisotropy.

    Returns:
        jax.Array: The blm parameters.
    """
    blmax = radiometer2blm_cache[0]
    lvals = radiometer2blm_cache[1]
    mvals = radiometer2blm_cache[2]
    gwtheta, gwphi = radiometer2blm_cache[3], radiometer2blm_cache[4]

    sqrt_map = jnp.sqrt(jnp.where(radiometer_map < 0, 0, radiometer_map)) # square root of amplitude after setting negatives to zero
    clms = jnp.dot(compute_ylm_maps(lvals,mvals,gwtheta,gwphi,blmax), sqrt_map) * 4*jnp.pi/768 # orthonormality condition of spherical harmonics

    # clms -> blms (complex and no negative m's) (Note that blms here are basically alms but at blmax rather than l_max)
    blms = jnp.ones( (blmax+1) + ((blmax+1)**2 - (blmax+1))//2, dtype=complex) # shape is n_(m=0) + n_(m>0) = blmax+1 + n_(m>0) = ...
    clm_0_index = np.argwhere(mvals == 0).flatten()
    l = jnp.concatenate([jnp.repeat(jnp.arange(blmax+1), jnp.array([ll + 1 for ll in range(blmax+1)]))])
    m = jnp.concatenate([jnp.arange(ll+1) for ll in range(blmax+1)])
    idx = m * (2 * blmax + 1 - m) // 2 + l
    blm = jnp.where( m == 0, clms[clm_0_index[l]],
                    (clms[clm_0_index[l] + m] + (-1)**m*1j*clms[clm_0_index[l] - m]) / jnp.sqrt(2) )
    blms = blms.at[idx].set(blm)

    # blms -> blm_params (split each blm into a real parameter and imaginary parameter)
    blm_params = jnp.empty((2*blms.size-2))
    blm_params = blm_params.at[0::2].set(blms[1:].real)
    blm_params = blm_params.at[1::2].set(blms[1:].imag) # the m = 0 imaginary components will be discarded in the residuals function
    
    blm_params = jnp.concatenate((jnp.array([0]), blm_params)) # add logA2
    
    return blm_params
