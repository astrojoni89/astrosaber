#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys
from configparser import ConfigParser

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata['package_name']
DESCRIPTION = metadata['description']
AUTHOR = metadata['author']
AUTHOR_EMAIL = metadata['author_email']
LICENSE = metadata['license']
URL = metadata['url']
__minimum_python_version__ = metadata["minimum_python_version"]

# Enforce Python version check - this is the same check as in __init__.py but
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    sys.stderr.write("ERROR: astrosaber requires Python {} or later\n".format(__minimum_python_version__))
    sys.exit(1)


from setuptools import setup, find_packages

readme_glob = 'README*'
with open(glob.glob(readme_glob)[0]) as f:
    LONG_DESCRIPTION = f.read()

# VERSION should be PEP440 compatible (http://www.python.org/dev/peps/pep-0440)
VERSION = metadata['version']

# Treat everything in scripts except README* as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
    if not os.path.basename(fname).startswith('README')]


setup(name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    scripts=scripts,
    install_requires=['astropy',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'tqdm',
                      'Sphinx',
                      'sphinx-autobuild',
                      'sphinx-autodoc-typehints',
                      'sphinx-rtd-theme',
                      'numpydoc',
                      ]
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    long_description=LONG_DESCRIPTION,
    python_requires='>={}'.format(__minimum_python_version__),
    packages=find_packages(),
)

