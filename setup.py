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
    # fast path (clebschGordan). POLICY [Claude fix]: numba is a HARD dependency --
    # default installs always get the fast builder. The try/except import guard in
    # clebschGordan exists only so an environment with a broken numba degrades
    # gracefully instead of breaking `import maps`: l_max<=63 is unaffected (exact
    # Python path), while l_max>=64 falls back to a pure-Python log-space builder
    # that is ~150x slower (numerically equivalent to ~1e-9, cached under a separate
    # key) and now emits a UserWarning so the degradation is never silent. The
    # clebschGordan.SAFE_LMAX guard bounds the high-l_max accuracy regime either way.
    # The previous (undeclared) 'enterprise' dependency is gone --
    # anis_coefficients is now vendored into maps/anis_coefficients.py.
    install_requires=['numpy', 'scipy', 'sympy', 'astroML', 'PTMCMCSampler', 'healpy', 'lmfit', 'numba'],
    # *strongly* suggested for sharing
    version='0.4.2',
    # The license can be anything you like
    license='MIT',
    description='Package to generate sky maps for PTA stochastic gravitational wave backgrounds.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
