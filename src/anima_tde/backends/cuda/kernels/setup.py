"""Build CUDA spiking ops extension."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tde_spiking_ops",
    ext_modules=[
        CUDAExtension(
            "tde_spiking_ops",
            ["spiking_ops.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_89,code=sm_89",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
