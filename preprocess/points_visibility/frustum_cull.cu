// -------------------------------------------------------------------
// frustum_cull.cu
// Fast GPU frustum culling for TLS point clouds.
//
// Two-kernel pipeline:
//   1. frustum_project_kernel  (N threads, one per world point)
//      - Transforms world → camera space
//      - Tests near/far depth and image bounds
//      - Atomically writes  (floatToOrderedInt(z) << 32 | orig_idx)
//        into combined_map[v*W+u] using atomicMin  →  nearest point wins
//
//   2. unpack_kernel  (H*W threads, one per pixel)
//      - Reads combined_map and fills depth_map (float32) + winner_map (int32)
//      - winner_map stores the ORIGINAL index into pts_world so the caller
//        can recover world coordinates without any auxiliary array:
//            vis_pts = pts_world[ winner_map[filtered_depth > 0] ]
//
// VRAM overhead: combined_map = H*W*8 bytes ≈ 12 MB for 1440×1080.
// -------------------------------------------------------------------

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// ── Bit-level float ↔ ordered-int helpers (identical to visibility_kernel.cu) ─

__device__ __forceinline__ int floatToOrderedInt(float f)
{
    int i = __float_as_int(f);
    return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat(int i)
{
    return __int_as_float((i >= 0) ? i : i ^ 0x7FFFFFFF);
}


// ── Kernel 1: project N world points and accumulate into combined_map ─────────

__global__ void frustum_project_kernel(
        const float* __restrict__ pts,      // (N, 3) float32
        const float* __restrict__ R,        // (9,)   float32, row-major world→cam
        const float* __restrict__ t,        // (3,)   float32
        float fx, float fy, float cx, float cy,
        int W, int H,
        float near_plane, float far_plane,
        unsigned long long* __restrict__ combined,  // (H*W,) init = 0xFFFF…
        int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = pts[3*i], y = pts[3*i+1], z = pts[3*i+2];

    // Transform to camera space
    float px = R[0]*x + R[1]*y + R[2]*z + t[0];
    float py = R[3]*x + R[4]*y + R[5]*z + t[1];
    float pz = R[6]*x + R[7]*y + R[8]*z + t[2];

    // Depth range test
    if (pz <= near_plane || pz >= far_plane) return;

    // Projection  (truncate toward zero, matches NumPy int32 cast)
    int u = (int)(px / pz * fx + cx);
    int v = (int)(py / pz * fy + cy);
    if (u < 0 || u >= W || v < 0 || v >= H) return;

    // Pack: high 32 bits = depth key (ordered int), low 32 bits = original idx
    // atomicMin → the smallest packed value (= nearest depth) survives per pixel
    unsigned long long val =
        ((unsigned long long)(unsigned int)floatToOrderedInt(pz) << 32)
        | (unsigned long long)(unsigned int)i;

    atomicMin(&combined[(long long)v * W + u], val);
}


// ── Kernel 2: unpack combined_map → depth_map + winner_map ───────────────────

__global__ void unpack_kernel(
        const unsigned long long* __restrict__ combined,
        float* __restrict__ depth_map,
        int*  __restrict__ winner_map,
        int HW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW) return;

    unsigned long long v = combined[i];
    if (v == 0xFFFFFFFFFFFFFFFFULL) {
        depth_map[i]  = 0.f;
        winner_map[i] = -1;
    } else {
        // Unpack depth and original index
        depth_map[i]  = orderedIntToFloat((int)(unsigned int)(v >> 32));
        winner_map[i] = (int)(unsigned int)(v & 0xFFFFFFFFULL);
    }
}


// ── Host wrapper ──────────────────────────────────────────────────────────────

std::vector<at::Tensor> frustum_cull_cuda(
        at::Tensor pts,   // (N, 3) float32 CUDA
        at::Tensor R_cw,  // (3,3) or (9,) float32 CUDA  — world→cam rotation
        at::Tensor t_cw,  // (3,) float32 CUDA
        float fx, float fy, float cx, float cy,
        int W, int H,
        float near_plane, float far_plane)
{
    const int N  = (int)pts.size(0);
    const int HW = H * W;

    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(pts.device());
    auto opts_i = at::TensorOptions().dtype(at::kInt).device(pts.device());
    auto opts_l = at::TensorOptions().dtype(at::kLong).device(pts.device());

    auto depth_map  = at::zeros({H, W}, opts_f);
    auto winner_map = at::full({H, W}, -1, opts_i);

    if (N == 0) return {depth_map, winner_map};

    // combined_map: all-ones bit pattern = "infinity / no winner"
    auto combined = at::full({HW}, (int64_t)-1LL, opts_l);

    const dim3 thr(256);

    // Kernel 1: project all N points
    frustum_project_kernel<<<(N + 255) / 256, thr>>>(
        pts.contiguous().data_ptr<float>(),
        R_cw.contiguous().view({-1}).data_ptr<float>(),
        t_cw.contiguous().data_ptr<float>(),
        fx, fy, cx, cy, W, H, near_plane, far_plane,
        (unsigned long long*)combined.data_ptr<int64_t>(),
        N);

    // Kernel 2: unpack combined_map
    unpack_kernel<<<(HW + 255) / 256, thr>>>(
        (unsigned long long*)combined.data_ptr<int64_t>(),
        depth_map.data_ptr<float>(),
        winner_map.data_ptr<int>(),
        HW);

    return {depth_map, winner_map};
}
