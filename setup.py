from setuptools import find_packages, setup

setup(
    name="pbmohpo",
    version="0.1",
    description="Preferential Bayesian Multi-objective Hyperparameter Optimization",
    url="https://github.com/ja-thomas/pbmohpo",
    author="Janek Thomas",
    author_email="janek.thomas@stat.uni-muenchen.de",
    license="LGPL",
    packages=find_packages(),
    install_requires=[
        "ConfigSpace",
        "botorch",
        "gpytorch",
        "torch",
        "yahpo-gym",
        "openml",
        "lightgbm",
        "matplotlib",
        "numpy",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort"],
        "experiments": ["yacs"],
        "test": ["pytest>=4.6"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
    },
)
