from setuptools import setup, find_packages


setup(
    name="pso4control",
    version="1.3.0",
    description="Lil package for studying applicability of PSO in control theory",
    author="Nikolai Makarov",
    author_email="makarov.nikolai20@gmail.com",
    packages=find_packages(),
    install_requires=[
        "click==8.1.7",
        "numpy==1.24.3",
        "seaborn==0.12.2",
        "sympy==1.12",
        "matplotlib==3.7.1",
    ],
    extras_requires={
        "log_parsing": ["jupyter", "pandas==2.0.2"]
    },
)
