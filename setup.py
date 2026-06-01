from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='maps',
    url='https://github.com/NihanPol/MAPS',
    author='Nihan Pol',
    author_email='nihan.pol@nanograv.org',
    # Needed to actually package something
    packages=['maps'],
    # Needed for dependencies
    # [Claude optimization] 'numba' added: it backs the high-l_max (>=64) calc_beta
    # fast path (clebschGordan). It is imported behind a guard; without it, l_max<=63
    # still works via the exact Python path, but l_max>=64 falls back to a pure-Python
    # log-space builder that is far slower and (like the numba path) loses accuracy to
    # Clebsch-Gordan cancellation as l_max grows -- the clebschGordan.SAFE_LMAX guard
    # bounds that regime either way. The previous (undeclared) 'enterprise' dependency
    # is gone -- anis_coefficients is now vendored into maps/anis_coefficients.py.
    install_requires=['numpy', 'scipy', 'sympy', 'astroML', 'PTMCMCSampler', 'healpy', 'lmfit', 'numba'],
    # *strongly* suggested for sharing
    version='0.4.2',
    # The license can be anything you like
    license='MIT',
    description='Package to generate sky maps for PTA stochastic gravitational wave backgrounds.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
