"""
setup.py — Build script for the C++17 AVX2 SIMD engine pybind11 extension.
Compiles with -O3 -mavx2 -mfma -std=c++17 on GCC/Clang,
or /O2 /arch:AVX2 /std:c++17 on MSVC.
"""
import os
import sys
import platform
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.0.0"

# Determine compiler flags based on platform
extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # MSVC flags
    extra_compile_args = ["/O2", "/arch:AVX2", "/std:c++17", "/EHsc"]
else:
    # GCC / Clang flags
    extra_compile_args = ["-O3", "-mavx2", "-mfma", "-std=c++17", "-ffast-math"]
    extra_link_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "simd_engine",
        ["simd_engine.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="simd_engine",
    version=__version__,
    author="SentinelAI Team",
    description="C++17 AVX2+FMA SIMD Vector Search Engine",
    long_description="High-performance cosine similarity, top-k search, and cache lookup using AVX2+FMA intrinsics.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
