from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            extra_compile_args={'nvcc': ['-O3', '--use_fast_math']},
        ),
        CUDAExtension(
            'frustum_cull_ext',
            [
                'frustum_cull/frustum_cull_bind.cpp',
                'frustum_cull/frustum_cull.cu',
            ],
            extra_compile_args={'nvcc': ['-O2']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
