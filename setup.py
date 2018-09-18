# -*- coding: utf-8 -*-
#
import codecs
import os
from setuptools import setup, find_packages

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "cplot", "__about__.py"), "rb") as f:
    exec(f.read(), about)


def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()


setup(
    name="cplot",
    packages=find_packages(),
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description="Plotting tools for complex-valued functions",
    long_description=read("README.md"),
    url="https://github.com/nschloe/cplot",
    license=about["__license__"],
    platforms="any",
    install_requires=["colorio", "matplotlib", "numpy"],
    long_description_content_type="text/markdown",
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
