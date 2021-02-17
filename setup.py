#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'astroSABER')
DESCRIPTION = metadata.get('description', 'astroSABER: Self-Absorption Baseline ExtractoR developed for systematic baseline fitting')
AUTHOR = metadata.get('author', 'Jonas Syed')
AUTHOR_EMAIL = metadata.get('author_email', 'syed [at] mpia-hd.mpg.de')
LICENSE = metadata.get('license', 'BSD 3-Clause')
URL = metadata.get('url', 'https://github.com/astrojoni89/astroSABER')
__minimum_python_version__ = metadata.get("minimum_python_version", "3.6")

# Enforce Python version check - this is the same check as in __init__.py but
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    sys.stderr.write("ERROR: astroSABER requires Python {} or later\n".format(__minimum_python_version__))
    sys.exit(1)


from setuptools import setup, find_packages

readme_glob = 'README*'
with open(glob.glob(readme_glob)[0]) as f:
    LONG_DESCRIPTION = f.read()

# VERSION should be PEP440 compatible (http://www.python.org/dev/peps/pep-0440)
VERSION = metadata.get('version', '1.0')

# Treat everything in scripts except README* as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
    if not os.path.basename(fname).startswith('README')]



setup(name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    scripts=scripts,
    install_requires=['astropy', 'numpy', 'scipy', 'tqdm'],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    long_description=LONG_DESCRIPTION,
    python_requires='>={}'.format(__minimum_python_version__),
    packages=find_packages(),
)

