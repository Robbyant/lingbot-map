#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> voxelize_frame_cuda(
    at::Tensor pts_xyz,
    at::Tensor pts_rgb,
    float voxel_size);

std::vector<at::Tensor> voxelize_frame(
    at::Tensor pts_xyz,
    at::Tensor pts_rgb,
    float voxel_size)
{
    TORCH_CHECK(pts_xyz.is_cuda(), "pts_xyz must be CUDA");
    TORCH_CHECK(pts_rgb.is_cuda(), "pts_rgb must be CUDA");
    return voxelize_frame_cuda(
        pts_xyz.contiguous(),
        pts_rgb.contiguous(),
        voxel_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_frame", &voxelize_frame, "GPU voxelization with Morton coding");
}
