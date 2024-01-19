import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name = 'valuingflexwater',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    version = '0.0.1',
    install_requires=required,
    include_package_data=True,
)
