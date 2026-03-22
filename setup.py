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
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Find CUTLASS
cutlass_path = os.environ.get("CUTLASS_PATH", "")
if not cutlass_path:
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
gencode_flags = []
for arch in gpu_archs.split(";"):
    arch_num = arch.replace(".", "")
    gencode_flags.extend(["-gencode", f"arch=compute_{arch_num},code=sm_{arch_num}"])

# Per-file optimal maxrregcount from sweep (B=10, L=4600, W=257, GH200):
#   D1 kernel:         reg=255 best (64.3ms vs 73.4ms at reg=128)
#   hdim32 fwd/bwd:    reg=128 best (D=2: 21.6 vs 44.4, D=4: 9.6 vs 15.8)
#   hdim64 fwd/bwd:    reg=255 best (1.1ms vs 2.8ms)
#   hdim128 fwd/bwd:   reg=128 best (0.9ms vs 1.1ms)
#   hdim256 fwd/bwd:   reg=128 default (untested at scale)
_PER_FILE_REGS = {
    "flash_d1_hdim1_bf16_sm80.cu": 255,
    "flash_dsmall_bf16_sm80.cu": 255,
    "flash_fwd_hdim32_bf16_sm80.cu": 128,
    "flash_bwd_hdim32_bf16_sm80.cu": 128,
    "flash_fwd_hdim64_bf16_sm80.cu": 255,
    "flash_bwd_hdim64_bf16_sm80.cu": 255,
    "flash_fwd_hdim128_bf16_sm80.cu": 128,
    "flash_bwd_hdim128_bf16_sm80.cu": 128,
    "flash_fwd_hdim256_bf16_sm80.cu": 128,
    "flash_bwd_hdim256_bf16_sm80.cu": 128,
}

_nvcc_base = [
    "-O3", "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--use_fast_math",
    "-lineinfo",
] + gencode_flags


class PerFileBuildExtension(BuildExtension):
    """Injects per-file --maxrregcount based on _PER_FILE_REGS."""

    def build_extensions(self):
        # Stash original _compile for monkey-patching
        original_compile = self.compiler._compile

        def patched_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            basename = os.path.basename(src)
            if basename in _PER_FILE_REGS and ext == ".cu":
                regs = _PER_FILE_REGS[basename]
                # Replace or add --maxrregcount for this file
                postargs = [a for a in extra_postargs if "--maxrregcount" not in a]
                postargs.append(f"--maxrregcount={regs}")
                return original_compile(obj, src, ext, cc_args, postargs, pp_opts)
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = patched_compile
        super().build_extensions()

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
                "flash_bias_attn/csrc/flash_d1_hdim1_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_dsmall_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_fwd_hdim32_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_bwd_hdim32_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_fwd_hdim64_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_bwd_hdim64_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_fwd_hdim128_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_bwd_hdim128_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_fwd_hdim256_bf16_sm80.cu",
                "flash_bias_attn/csrc/flash_bwd_hdim256_bf16_sm80.cu",
            ],
            include_dirs=[
                os.path.abspath("flash_bias_attn/csrc"),
                cutlass_include,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": _nvcc_base + ["--maxrregcount=128"],  # default, overridden per file
            },
        )
    ],
    cmdclass={"build_ext": PerFileBuildExtension},
)
