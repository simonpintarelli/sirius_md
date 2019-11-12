from setuptools import setup

setup(
    name='velocity_verlet',
    version='0.5',
    author='Simon Pintarelli <simon.pintarelli@cscs.ch>',
    packages=['sirius_md'],
    description='Wavefunction extrapolation for ab initio MD',
    install_requires=['pyyaml', 'numpy', 'matplotlib'],
    entry_points={'console_scripts':
                  'velocity_verlet = sirius_md.verlet:run'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
