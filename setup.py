# Distributed under the MIT License.
# See LICENSE for details.

from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='nbpy',
    version='0.1.0',
    description="Recreational N-body problem solver.",
    long_description=long_description,
    url='https://github.com/carmaza/nbpy',
    author='Cristóbal Armaza',
    author_email='ca455@cornell.edu',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    keywords='astrophysics galactic n-body newton gravity',
    packages=find_packages(),
    install_requires=['matplotlib>=3.5.1', 'numpy>=1.22.0', 'scipy>=1.8.0'],
    python_requires='>=3.8')
