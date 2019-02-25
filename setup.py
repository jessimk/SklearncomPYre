#!/usr/bin/env python
from distutils.core import setup

setup(
    name='SklearncomPYre',
    version='v0.0',
    packages=['SklearncomPYre'],
	url = ['https://github.com/UBC-MDS/SklearncomPYre'],
    license='MIT',
	long_description=open('README.md').read(),
	install_requires=['numpy','sklearn','pandas','matplotlib']
)
