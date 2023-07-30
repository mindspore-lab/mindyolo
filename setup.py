#!/usr/bin/env python
import os.path
import subprocess
import pathlib
import sys
import glob

from setuptools import find_packages, setup

exec(open("mindyolo/version.py").read())

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def parse_requirements(path=here / "requirements.txt"):
    """parse requirements in file"""
    pkgs = []
    if not os.path.exists(path):
        return pkgs
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith("#"):
                continue
            pkgs.append(line)
    return pkgs


def compile_fused_op(path=here / "mindyolo/models/losses/fused_op"):
    try:
        check_txt = subprocess.run(["nvcc", "--version"], timeout=3, capture_output=True, check=True).stdout
        if "command not found" in str(check_txt):
            print("nvcc not found, skipped compiling fused operator.")
            return
        for fused_op_src in glob.glob(str(path / "*_kernel.cu")):
            fused_op_so = f"{fused_op_src[:-3]}.so"
            so_path = str(path / fused_op_so)
            nvcc_cmd = "nvcc --shared -Xcompiler -fPIC -o " + so_path + " " + fused_op_src
            print("nvcc compiler cmd: {}".format(nvcc_cmd))
            os.system(nvcc_cmd)
    except FileNotFoundError:
        print("nvcc not found, skipped compiling fused operator.")
        return
    except subprocess.CalledProcessError as e:
        print("nvcc execute failed, skipped compiling fused operator: ", e)
        return


# add c++ extension
ext_modules = []
try:
    from pybind11.setup_helpers import Pybind11Extension
    args = ["-O2"] if sys.platform == "win32" else ["-O3", "-std=c++14", "-g", "-Wno-reorder"]
    ext_modules = [
        Pybind11Extension(
            name="mindyolo.csrc.fast_coco_eval.fast_coco_eval",  # use relative path
            sources=["mindyolo/csrc/fast_coco_eval/cocoeval/cocoeval.cpp"],  # use relative path
            include_dirs=["mindyolo/csrc/fast_coco_eval/cocoeval"],  # use relative path
            extra_compile_args=args
        ),
    ]
except ImportError:
    pass
compile_fused_op()
setup(
    name="mindyolo",
    author="MindSpore Ecosystem",
    author_email="mindspore-ecosystem@example.com",
    url="https://github.com/mindspore-lab/mindyolo",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindyolo",
        "Issue Tracker": "https://github.com/mindspore-lab/mindyolo/issues",
    },
    description="A toolbox of vision models and algorithms based on MindSpore.",
    long_description=long_description,
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["mindyolo", "mindyolo.*"]),
    package_data={"mindyolo": ["models/losses/fused_op/*_kernel.so"]},
    install_requires=parse_requirements(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    ext_modules=ext_modules,
    zip_safe=False,
)
