from setuptools import find_packages


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="flowvision",
    version="0.0.2",  # 版本号
    author="flow vision contributors",
    author_email="596106517@qq.com",
    description="oneflow vision codebase",
    license="MIT",
    packages=find_packages(),  # 需要安装的代码包，也可以用find_packages函数
    install_requires=["rich",],
    classifiers=[
        "License :: OSI Approved :: MIT License",
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
