#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint64_t morton3D(int x, int y, int z) {
    auto expandBits = [](uint64_t v) -> uint64_t {
        v = (v | (v << 32)) & 0x1f00000000ffffULL;
        v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
        v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2)) & 0x1249249249249249ULL;
        return v;
    };
    uint64_t xx = expandBits((uint64_t)(uint32_t)x);
    uint64_t yy = expandBits((uint64_t)(uint32_t)y);
    uint64_t zz = expandBits((uint64_t)(uint32_t)z);
    return xx | (yy << 1) | (zz << 2);
}

__global__ void compute_morton_kernel(
    const float* __restrict__ pts_xyz,
    const float* __restrict__ pts_rgb,
    float voxel_size,
    int N,
    int64_t* __restrict__ morton_codes,
    float* __restrict__ voxel_centers,
    float* __restrict__ voxel_colors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int ix = (int)floorf(pts_xyz[3 * i + 0] / voxel_size);
    int iy = (int)floorf(pts_xyz[3 * i + 1] / voxel_size);
    int iz = (int)floorf(pts_xyz[3 * i + 2] / voxel_size);

    const int OFFSET = (1 << 20);
    uint64_t code = morton3D(ix + OFFSET, iy + OFFSET, iz + OFFSET);
    morton_codes[i] = (int64_t)code;

    voxel_centers[3 * i + 0] = (ix + 0.5f) * voxel_size;
    voxel_centers[3 * i + 1] = (iy + 0.5f) * voxel_size;
    voxel_centers[3 * i + 2] = (iz + 0.5f) * voxel_size;

    voxel_colors[3 * i + 0] = pts_rgb[3 * i + 0];
    voxel_colors[3 * i + 1] = pts_rgb[3 * i + 1];
    voxel_colors[3 * i + 2] = pts_rgb[3 * i + 2];
}

__global__ void mark_unique_kernel(
    const int64_t* __restrict__ sorted_morton,
    int N,
    bool* __restrict__ unique_mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i == 0) {
        unique_mask[i] = true;
    } else {
        unique_mask[i] = (sorted_morton[i] != sorted_morton[i - 1]);
    }
}

std::vector<at::Tensor> voxelize_frame_cuda(
    at::Tensor pts_xyz,
    at::Tensor pts_rgb,
    float voxel_size)
{
    int N = pts_xyz.size(0);
    if (N == 0) {
        auto opts_f = at::TensorOptions().dtype(at::kFloat).device(pts_xyz.device());
        auto opts_l = at::TensorOptions().dtype(at::kLong).device(pts_xyz.device());
        return {
            at::zeros({0, 3}, opts_f),
            at::zeros({0, 3}, opts_f),
            at::zeros({0}, opts_l)
        };
    }

    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(pts_xyz.device());
    auto opts_l = at::TensorOptions().dtype(at::kLong).device(pts_xyz.device());
    auto opts_b = at::TensorOptions().dtype(at::kBool).device(pts_xyz.device());

    auto morton_codes = at::empty({N}, opts_l);
    auto voxel_centers = at::empty({N, 3}, opts_f);
    auto voxel_colors = at::empty({N, 3}, opts_f);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_morton_kernel<<<blocks, threads>>>(
        pts_xyz.data_ptr<float>(),
        pts_rgb.data_ptr<float>(),
        voxel_size, N,
        morton_codes.data_ptr<int64_t>(),
        voxel_centers.data_ptr<float>(),
        voxel_colors.data_ptr<float>());

    auto sorted_result = at::sort(morton_codes);
    auto sorted_morton = std::get<0>(sorted_result);
    auto sort_indices = std::get<1>(sorted_result);

    voxel_centers = voxel_centers.index_select(0, sort_indices);
    voxel_colors = voxel_colors.index_select(0, sort_indices);

    auto unique_mask = at::empty({N}, opts_b);
    mark_unique_kernel<<<blocks, threads>>>(
        sorted_morton.data_ptr<int64_t>(),
        N,
        unique_mask.data_ptr<bool>());

    auto unique_indices = at::nonzero(unique_mask).squeeze(1);
    auto unique_centers = voxel_centers.index_select(0, unique_indices);
    auto unique_colors = voxel_colors.index_select(0, unique_indices);
    auto unique_morton = sorted_morton.index_select(0, unique_indices);

    return {unique_centers, unique_colors, unique_morton};
}
