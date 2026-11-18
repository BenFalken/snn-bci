from setuptools import setup, find_packages

#installed into the same environment as NeuralDecoder (whose torch/numpy pins it uses), so rnn_step3 can run both decoders
setup(
    name='snn_decoder',
    version='0.0.1',
    packages=find_packages(include=['snnDecoder', 'snnDecoder.*']),
    package_data={'snnDecoder': ['configs/*.yaml']},
    install_requires=[
        'torch>=1.8',
        'sinabs>=1.2',
        'numpy',
        'scipy',
        'pyyaml',
        'matplotlib',
        'psutil',
    ]
)
