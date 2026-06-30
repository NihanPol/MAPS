# MApping the PTA sky (MAPS)

A PTA community developed and maintained Python package that aims to simplify the task of searching for anisotropies in the nanohertz gravitational wave sky observed with PTAs. 

## Installation

As of now, the easiest way to install MAPS is to use pip+git:

For this method you MUST install ENTERPRISE through conda first. We recommend installing it with enterprise_extensions for its additional utility and will install ENTERPRISE along with it:

```bash
conda install -c conda-forge enterprise_extensions
pip install git+https://github.com/NihanPol/MAPS
```

This has had the fewest problems and is likely the easiest way to get MAPS working.


## Usage

While there are a few examples for using MAPS found in the maps/docs folder, anis_pta.py has been overhauled to be more explicit in its documentation. If you are using VSCode or Jupyter notebooks, the doc comments should appear when attempting to use a command. If things are not clear, that likely isn't your fault, and raising a github issue with the problem will bring our attention to it and fix it!

