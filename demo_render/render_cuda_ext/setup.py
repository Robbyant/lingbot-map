import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if sys.platform == 'win32':
    # MSVC defaults to a non-conforming preprocessor and an older C++
    # standard, which chokes on pybind11/torch's macro-heavy headers.
    # Neither flag is needed (or understood) by gcc/clang, so keep this
    # Windows-only rather than passing it to every platform's compiler.
    win_cxx_args = ['/std:c++20', '/Zc:preprocessor']
    win_nvcc_args = ['-std=c++20', '-Xcompiler=/Zc:preprocessor']
else:
    win_cxx_args = []
    win_nvcc_args = []

setup(
    name='render_cuda_ext',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'voxel_morton_ext',
            [
                'voxel_morton/voxel_morton_bind.cpp',
                'voxel_morton/voxel_morton.cu',
            ],
            extra_compile_args={
                'cxx': win_cxx_args,
                'nvcc': ['-O3', '--use_fast_math'] + win_nvcc_args,
            },
        ),
        CUDAExtension(
            'frustum_cull_ext',
            [
                'frustum_cull/frustum_cull_bind.cpp',
                'frustum_cull/frustum_cull.cu',
            ],
            extra_compile_args={
                'cxx': win_cxx_args,
                'nvcc': ['-O2'] + win_nvcc_args,
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)