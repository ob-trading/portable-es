import io
import setuptools

setuptools.setup(
    name='portable-es',
    version='1.2.14',
    description=(
        'Portable ES is a distributed gradient-less optimization framework built on PyTorch & Numpy.'
    ),
    author='Casper',
    author_email='casper@devdroplets.com',
    url='https://github.com/ob-trading/portable-es',
    license='MPL2',
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=['portable_es'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'torch>=1.4.0',
        'numpy',
        'tensorboardx',
        'distributed_worker>=1.2.1'
    ],
)