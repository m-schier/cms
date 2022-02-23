from setuptools import setup, find_packages

setup(
    name='CMS',
    author='Maximilian Schier, Christoph Reinders',
    version='1.0',
    packages=find_packages(exclude=('Util',)),
    install_requires=['numpy', 'numba', 'scipy'],
    extras_require={'eval': ['sklearn', 'h5py'], 'train': ['tensorflow', 'tqdm', 'opencv-python']}
)
