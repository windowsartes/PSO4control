from setuptools import setup, find_packages


setup(
    name="pso4control",
    version="2.0.1",
    description="Lil package for studying applicability of PSO in control theory",
    author="Nikolai Makarov",
    author_email="makarov.nikolai20@gmail.com",
    packages=find_packages(),
    install_requires=[
        "click==8.1.7",
        "matplotlib==3.9.1",
        "numpy==2.0.1",
        "pydantic==2.8.2",
        "seaborn==0.13.2",
        "sympy==1.13.1",
        "scipy",
    ]
)
