#!/usr/bin/env python
from setuptools import setup, Extension
import os, sys

setup(
        name='pydem',
        version='0.0.1',
        description='Particle interaction',
        author='Denis Demidov',
        author_email='dennis.demidov@gmail.com',
        license='MIT',
        include_package_data=True,
        zip_safe=False,
        packages=['pydem'],
        ext_modules=[
            Extension('pydem.pydem_ext', ['pydem/pydem.cpp'],
                include_dirs=['pydem', 'pybind11/include', 'vexcl'],
                libraries=['boost_system', 'boost_filesystem', 'OpenCL'],
                extra_compile_args=['-O3', '-std=c++11', '-flto',
                    '-Wno-deprecated-declarations', '-Wno-sign-compare',
                    '-Wno-ignored-attributes']
                )
            ]
)
