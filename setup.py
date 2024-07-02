"""
The code is modfied from
https://github.com/state-spaces/mamba
"""
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "quamba"

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
    return nvcc_extra_args + ["--threads", max_jobs]


cmdclass = {}
ext_modules = []

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

check_if_cuda_home_none(PACKAGE_NAME)
# Check, if CUDA11 is installed for compute capability 8.0
cc_flag = []
if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.6"):
        raise RuntimeError(
            f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
            "Note: make sure nvcc has a supported version by running nvcc -V."
        )
# DO NOT USE GENCODE FLAG FOR CUTLASS
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_80,code=sm_80")
# cc_flag.append("-arch=sm_80")

ext_modules.append(
    CUDAExtension(
        name="quant_sscan_cuda",
        sources=[
            "csrc/selective_scan/quant_sscan.cpp",
            "csrc/selective_scan/quant_sscan_fwd.cu",
            "csrc/selective_scan/quant_sscan_update.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "selective_scan",
        ],
    )
)


# DO NOT USE GENCODE FLAG FOR CUTLASS
ext_modules.append(
    CUDAExtension(
        name="quant_linear_cuda",
        sources=[
            "csrc/linear/quant_linear.cpp",
            "csrc/linear/quant_linear_fwd.cu",
        ],
        extra_link_args=['-lcublas_static', '-lcublasLt_static',
                            '-lculibos', '-lcudart', '-lcudart_static',
                            '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    f"-DCUDA_ARCH={cuda_arch}",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF2_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "linear",
            Path(this_dir) / "3rdparty" / "cutlass" / "build" / "install" / "include",
        ],
    )
)


ext_modules.append(
    CUDAExtension(
        name="quant_hadamard",
        sources=[
            "csrc/hadamard/quant_hadamard.cpp",
            "csrc/hadamard/quant_hadamard_fwd.cu",
        ],
        extra_link_args=['-lcublas_static', '-lcublasLt_static',
                            '-lculibos', '-lcudart', '-lcudart_static',
                            '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    f"-DCUDA_ARCH={cuda_arch}",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF2_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "hadamard",
        ],
    )
)


ext_modules.append(
    CUDAExtension(
        name="quant_causal_conv1d_cuda",
        sources=[
            "csrc/causal_conv1d/quant_causal_conv1d.cpp",
            "csrc/causal_conv1d/quant_causal_conv1d_fwd.cu",
            "csrc/causal_conv1d/quant_causal_conv1d_update.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "causal_conv1d",
        ],
    )
)

ext_modules.append(
    CUDAExtension(
        name="rms_norm_cuda",
        sources=[
            "csrc/norm/rms_norm.cpp",
            "csrc/norm/rms_norm_fwd.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "norm",
        ],
    )
)


def get_package_version():
    with open(Path(this_dir) / "quamba" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("Q_MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "mamba_ssm.egg-info",
        )
    ),
    author="Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Diana Marculescu",
    author_email="hungyueh.chiang@utexas.edu",
    description="Quamba: Post-training Quantization for Selective State Space Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enyac-group/Quamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": bdist_wheel, "build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        "triton",
        "transformers",
        "causal_conv1d>=1.1.0",
        "fast_hadamard_transform"
    ],
)
