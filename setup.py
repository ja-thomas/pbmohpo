from setuptools import setup, find_packages

setup(
    name="pbmohpo",
    version="0.1",
    description="Preferential Bayesian Multi-objective Hyperparameter Optimization",
    url="https://github.com/ja-thomas/pbmohpo",
    author="Janek Thomas",
    author_email="janek.thomas@stat.uni-muenchen.de",
    license="LGPL",
    packages=find_packages(),
    install_requires=["configspace", "botorch"],
)
