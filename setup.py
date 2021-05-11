from setuptools import setup, find_packages

setup(
    name='flatland-temperatureobservation',
    version='0.0.1',
    install_requires=["numpy", "flatland-rl", "scipy"],
    packages=find_packages()
)
