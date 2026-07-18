#include <torch/extension.h>

at::Tensor frustum_cull_cuda(
    at::Tensor pts, at::Tensor R_cw, at::Tensor t_cw,
    float fx, float fy, float cx, float cy,
    int W, int H, float near_plane, float far_plane);

at::Tensor frustum_cull(
    at::Tensor pts, at::Tensor R_cw, at::Tensor t_cw,
    float fx, float fy, float cx, float cy,
    int W, int H, float near_plane, float far_plane)
{
    TORCH_CHECK(pts.is_cuda(), "pts must be CUDA");
    TORCH_CHECK(R_cw.is_cuda(), "R_cw must be CUDA");
    TORCH_CHECK(t_cw.is_cuda(), "t_cw must be CUDA");
    return frustum_cull_cuda(
        pts.contiguous(), R_cw.contiguous(), t_cw.contiguous(),
        fx, fy, cx, cy, W, H, near_plane, far_plane);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frustum_cull", &frustum_cull, "GPU frustum culling -> visible bool mask (N,)");
}
