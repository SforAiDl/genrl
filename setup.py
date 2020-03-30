import os

from setuptools import setup, find_packages

## Basic information
NAME = "jigglypuff-rl"
DESCRIPTION = "PyTorch implementations of reinforcement learning algorithms."
AUTHOR = "Society for Artificial Intelligence and Deep Learning"
EMAIL = "ajaysubramanian110@gmail.com"
LICENSE = "MIT"
REPOSITORY = "https://github.com/SforAiDl/JigglypuffRL"
PACKAGE = "JigglypuffRL"
with open('README.md','r') as f:
    LONG_DESCRIPTION = f.read()

## Define the keywords
KEYWORDS = (
    "reinforcement learning",
    "pytorch",
    "neuroscience",
    "machine learning"
)

## Define the classifiers
## See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS  = (
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
)

## Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
VERSION_PATH = os.path.join(PACKAGE, "version.py")
PKG_DESCRIBE = "DESCRIPTION.md"

## Directories to ignore in find_packages
EXCLUDES = ()

## Define the configuration
config = {
    "name": NAME,
    "version": '0.0.1',
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": 'text/markdown',
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "url": REPOSITORY,
    "project_urls": {
        "Source": REPOSITORY,
    },
    "packages": find_packages(where=PROJECT, exclude=EXCLUDES),
    "python_requires": ">=3.6",
}

if __name__ == '__main__':
    setup(**config)