import copy
import os
import glob
import setuptools
import subprocess
import distutils.command.clean
import distutils.spawn

# import oneflow as flow
from setuptools import find_packages
from setuptools.command.build_ext import build_ext


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

version = "0.0.55"
package_name = "flowvision"
cwd = os.path.dirname(os.path.abspath(__file__))

sha = "Unknown"
try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass


def write_version_file():
    version_path = os.path.join(cwd, "flowvision", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in [".cu", ".cuh"]


def _find_cuda_home():
    """
    Finds the CUDA install path.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            nvcc = subprocess.check_output(["which", "nvcc"]).decode().rstrip("\r\n")
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not flow.cuda.is_available():
        print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    return cuda_home


def _get_oneflow_include_path():
    from oneflow.framework.sysconfig import get_include

    return get_include()


def _get_oneflow_compile_flags():
    from oneflow.framework.sysconfig import get_compile_flags

    return get_compile_flags()


def _get_oneflow_lib_path():
    from oneflow.framework.sysconfig import get_lib

    lib_path = glob.glob(os.path.join(get_lib(), "_oneflow_internal.*.so"))
    return lib_path


def _join_cuda_home(*paths):
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        raise EnvironmentError(
            "cuda_home environment variable is not set. "
            "Please set it to your CUDA install root."
        )
    return os.path.join(cuda_home, *paths)


def CppExtension(name, sources, *args, **kwargs):
    """
    Creates a :class:`setuptools.Extension` for C++.
    """
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs.append(_get_oneflow_include_path())
    kwargs["include_dirs"] = include_dirs
    kwargs["language"] = "c++"
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Creates a :class:`setuptools.Extension` for CUDA/C++.
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += _get_oneflow_lib_path()
    library_dirs.append("{}/lib64".format(_find_cuda_home()))
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("cudart")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs.append(_get_oneflow_include_path())
    include_dirs.append("{}/include".format(_find_cuda_home()))
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext, object):
    """
    A custom :mod:`setuptools` build extension.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns an alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        def init_with_options(*args, **kwargs):
            kwargs = kwargs.copy()
            kwargs.update(options)
            return cls(*args, **kwargs)

        return init_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def build_extensions(self):
        for extension in self.extensions:
            self._add_compile_flag(extension, _get_oneflow_compile_flags())
            self._define_oneflow_extension_name(extension)

        self.compiler.src_extensions += [".cu", ".cuh"]
        original_compile = self.compiler._compile

        def unix_warp_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home("bin", "nvcc")
                    if not isinstance(nvcc, list):
                        nvcc = [nvcc]
                    self.compiler.set_executable("compiler_so", nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["nvcc"][0]
                    cflags = [
                        "--compiler-options",
                        "-fPIC",
                        "-O3",
                        "-Xcompiler",
                        "-Wextra",
                        "--disable-warnings",
                        "-DWITH_CUDA",
                    ]
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"][0]
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith("-std=") for flag in cflags):
                    cflags.append("-std=c++14")

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        self.compiler._compile = unix_warp_compile
        build_ext.build_extensions(self)

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_oneflow_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like oneflow._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = "-DONEFLOW_EXTENSION_NAME={}".format(name)
        self._add_compile_flag(extension, define)


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "flowvision", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "ops", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "ops", "*.cu"))

    sources = main_file
    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": []}
    cuda_home = _find_cuda_home()
    if flow.cuda.is_available() and (cuda_home is not None):
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "flowvision._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_objects=_get_oneflow_lib_path(),
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("README.md", "r") as fh:
        long_description = fh.read()

    write_version_file()

    setup(
        name=package_name,
        version=version,  # version number
        author="flow vision contributors",
        author_email="rentianhe@oneflow.org",
        description="oneflow vision codebase",
        license="BSD",
        packages=find_packages(),
        install_requires=["rich", "tabulate", "six",],
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
