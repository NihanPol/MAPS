from setuptools import setup
import pathlib


# Helper to read requirements from a file
def parse_requirements(filename):
    with open(pathlib.Path(__file__).parent / filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


# Extra requires for bilby samplers
bb_sampler_requirements = parse_requirements("bilby_sampler_requirements.txt")


setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='maps',
    url='https://github.com/NihanPol/MAPS',
    author='Nihan Pol',
    author_email='nihan.pol@nanograv.org',
    # Needed to actually package something
    packages=['maps'],
    # Needed for dependencies: 
    # Order -> Frequentist + Bayesian ('enterprise-pulsar' for priors in PTMCMC setup) + bilby samplers
    install_requires=['numpy', 'scipy', 'sympy', 'astroML', 'healpy', 'lmfit'] + ['PTMCMCSampler', 'enterprise-pulsar', 'bilby', 'corner', 'la-forge'] + bb_sampler_requirements,
    # *strongly* suggested for sharing
    version='0.4.3',
    # The license can be anything you like
    license='MIT',
    description='Package to generate sky maps for PTA stochastic gravitational wave backgrounds.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
