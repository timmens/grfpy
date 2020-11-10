from setuptools import find_packages  # noqa: D100
from setuptools import setup

setup(
    name="grfpy",
    version="0.0.0dev1",
    description="Python wrapper for the R package: grf.",
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Development Status :: Pre-alpha",
    ],
    keywords=["causal inference", "machine learning"],
    url="https://github.com/timmens/grfpy",
    author="Tim Mensinger",
    author_email="mensingertim@gmail.com",
    packages=find_packages(exclude=["tests/*"]),
    zip_safe=False,
)
