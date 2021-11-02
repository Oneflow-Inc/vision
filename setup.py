from setuptools import find_packages


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flowvision",
    version="0.0.4",  # version number
    author="flow vision contributors",
    author_email="596106517@qq.com",
    description="oneflow vision codebase",
    license="BSD",
    packages=find_packages(),
    install_requires=["rich",],
    classifiers=[
        "License :: OSI Approved :: BSD 3-Clause License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
    keywords="computer vision",
    url="https://github.com/Oneflow-Inc/vision",
    platforms="any",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
