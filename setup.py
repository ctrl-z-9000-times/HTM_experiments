#!/usr/bin/python3
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'HTM_experiments',
  ext_modules = cythonize("sdr.pyx"),
)
