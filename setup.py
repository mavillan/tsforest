from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='tsforest',
    version='0.2.54',
    author='Martín Villanueva',
    author_email='nallivam@gmail.com',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    license='GPLv3',
)
