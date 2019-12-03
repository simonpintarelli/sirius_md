from setuptools import setup

setup(
    name='velocity_verlet',
    version='0.5',
    packages=['sirius_md'],
    description='Wavefunction extrapolation for ab initio MD',
    install_requires=['pyyaml', 'numpy', 'matplotlib'],
    scripts=['bin/verlet'],
    entry_points={'console_scripts':
                  ['run_verlet = sirius_md.verlet:run',
                   'replay_verlet = sirius_md.verlet_replay:run']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
