"""
Build flash-bias-attn CUDA extension.

Requirements:
  - PyTorch with CUDA support
  - CUTLASS (git clone https://github.com/NVIDIA/cutlass.git)
  - CUDA toolkit

Build:
  CC=gcc CXX=g++ CUTLASS_PATH=/path/to/cutlass python setup.py build_ext --inplace

Or with pip:
  CC=gcc CXX=g++ CUTLASS_PATH=/path/to/cutlass pip install .
"""
import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Find CUTLASS
cutlass_path = os.environ.get("CUTLASS_PATH", "")
if not cutlass_path:
    # Try common locations
    for candidate in [
        os.path.expanduser("~/cutlass"),
        "/usr/local/cutlass",
        "../cutlass",
        "cutlass",
    ]:
        if os.path.isdir(os.path.join(candidate, "include", "cutlass")):
            cutlass_path = candidate
            break

if cutlass_path:
    cutlass_include = os.path.join(cutlass_path, "include")
else:
    print("WARNING: CUTLASS not found. Set CUTLASS_PATH env var.")
    print("  git clone https://github.com/NVIDIA/cutlass.git")
    cutlass_include = "cutlass/include"

print(f"CUTLASS include: {cutlass_include}")

# Detect GPU architectures
gpu_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;9.0").replace(" ", "")
max_rregcount = os.environ.get("FLASH_BIAS_MAXRREGCOUNT", "128")
gencode_flags = []
for arch in gpu_archs.split(";"):
    arch_num = arch.replace(".", "")
    gencode_flags.extend(["-gencode", f"arch=compute_{arch_num},code=sm_{arch_num}"])

setup(
    name="flash-bias-attn",
    version="0.1.0",
    description="Flash Attention with exact per-head relative position bias",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Jian Zhou",
    url="https://github.com/jzthree/flash-bias-attn",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["torch>=2.0", "flash-attn>=2.0"],
    ext_modules=[
        CUDAExtension(
            name="flash_attn_bias_cuda",
            sources=[
                "flash_bias_attn/csrc/flash_api_bias.cpp",
                "flash_bias_attn/csrc/flash_fwd_hdim32_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_bwd_hdim32_bf16_sm80.cu",
            ],
            include_dirs=[
                os.path.abspath("flash_bias_attn/csrc"),
                cutlass_include,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--use_fast_math", f"--maxrregcount={max_rregcount}",
                    "-lineinfo",
                ] + gencode_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
