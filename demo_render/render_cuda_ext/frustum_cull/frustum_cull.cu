#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void frustum_cull_kernel(
    const float* __restrict__ pts,
    const float* __restrict__ R,
    const float* __restrict__ t,
    float fx, float fy, float cx, float cy,
    int W, int H,
    float near_plane, float far_plane,
    bool* __restrict__ visible,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = pts[3 * i];
    float y = pts[3 * i + 1];
    float z = pts[3 * i + 2];
    float px = R[0] * x + R[1] * y + R[2] * z + t[0];
    float py = R[3] * x + R[4] * y + R[5] * z + t[1];
    float pz = R[6] * x + R[7] * y + R[8] * z + t[2];

    if (pz <= near_plane || pz >= far_plane) {
        visible[i] = false;
        return;
    }

    int u = (int)(px / pz * fx + cx);
    int v = (int)(py / pz * fy + cy);
    visible[i] = (u >= 0 && u < W && v >= 0 && v < H);
}

at::Tensor frustum_cull_cuda(
    at::Tensor pts, at::Tensor R_cw, at::Tensor t_cw,
    float fx, float fy, float cx, float cy,
    int W, int H, float near_plane, float far_plane)
{
    const int N = (int)pts.size(0);
    auto opts_b = at::TensorOptions().dtype(at::kBool).device(pts.device());
    auto visible = at::zeros({N}, opts_b);
    if (N == 0) return visible;

    frustum_cull_kernel<<<(N + 255) / 256, 256>>>(
        pts.contiguous().data_ptr<float>(),
        R_cw.contiguous().view({-1}).data_ptr<float>(),
        t_cw.contiguous().data_ptr<float>(),
        fx, fy, cx, cy, W, H, near_plane, far_plane,
        visible.data_ptr<bool>(), N);

    return visible;
}
