
// -------------------------------------------------------------------
// Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
// Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
// Released under Creative Commons
// Attribution-NonCommercial-ShareAlike 4.0 International License.
// http://creativecommons.org/licenses/by-nc-sa/4.0/
// -------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

    #define IMAGE_SIZE 10000
    #define TILE_W 25	//Tile Width
    #define TILE_H 25	//Tile Height
    #define R 3		//Filter Radius
    #define D (2*R+1)	//Filter Diameter
    #define BLOCK_W (TILE_W+(2*R))
    #define BLOCK_H (TILE_H+(2*R))


__device__ __forceinline__ int floatToOrderedInt( float floatVal ) {
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( int intVal ) {
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__global__ void depth_image_kernel(
    const int* __restrict__ in_uv, 
    const float* __restrict__ in_depth,
    float* __restrict__  out,
    unsigned int size, unsigned int width, unsigned int height) {

        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index >= 0 && index < size) {
            int uv_index = in_uv[2*index+1]*width + in_uv[2*index];
            atomicMin((int *)&out[uv_index], floatToOrderedInt(in_depth[index]));
        }

    }

template <typename scalar_t>
__global__ void visibility_kernel(
    const scalar_t* __restrict__ in, 
    scalar_t* __restrict__  out, 
    unsigned int width, unsigned int height, unsigned int threshold) {
    //__shared__ scalar_t sharedImg[BLOCK_W*BLOCK_H];
    //int x = blockIdx.x*TILE_W + threadIdx.x - R;
    //int y = blockIdx.y*TILE_H + threadIdx.y - R;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, height-1);
    y = max(y, 0);
    y = min(y, width-1);
    //unsigned int index = y*width + x;
    unsigned int index = x*width + y;
    //unsigned int bindex = threadIdx.y*blockDim.y+threadIdx.x;

    // each thread copy its pixel to the shared memory
    //sharedImg[bindex] = in[index];
    //__syncthreads();

    //index = 

    //out[index] = 7.0;
    //return;

    // only threads inside the apron write the results
    //if ((threadIdx.x >= R) && (threadIdx.x < (BLOCK_W-R)) &&
    //        (threadIdx.y >= R) && (threadIdx.y < (BLOCK_H-R))) {
    out[index] = in[index];
    if(x >= R && y >= R && x < (height-R) && y < (width-R)) {
        scalar_t pixel = in[index];
        if (pixel != 0.)  {
            int sum = 0;
            int count = 0;
            for(int i=-R; i<=R; i++) {
                for(int j=-R; j<=R; j++) {
                    if(i==0 && j==0)
                        continue;
                    int temp_index = index + i*width + j;
                    scalar_t temp_pixel = in[temp_index];
                    if(temp_pixel != 0 ) {
                        count += 1;
                        if(temp_pixel < pixel - 3.)
                            sum += 1;
                    }
                }
            }
            if(sum >= 1+threshold * count / (R*R*2*2))
                out[index] = 0.;
        }

    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t norm(scalar_t x, scalar_t y, scalar_t z) {
    scalar_t sum = x*x + y*y + z*z;
    return sum;
}


template <typename scalar_t>
__global__ void visibility_kernel2(
    const scalar_t* __restrict__ in_depth,
    const scalar_t* __restrict__ intrinsic,
    scalar_t* __restrict__  out,
    unsigned int width, unsigned int height, float threshold, unsigned int radius) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, height-1);
    y = max(y, 0);
    y = min(y, width-1);

    //unsigned int index = y*width + x;
    unsigned int index = x*width + y;
    out[index] = in_depth[index];

    bool debug=false;

    if(x >= 0 && y >= 0 && x < (height) && y < (width)) {

        scalar_t pixel = in_depth[index];
        if (pixel != 0.)  {
            scalar_t fx, fy, cx, cy;
            fx = intrinsic[0];
            fy = intrinsic[1];
            cx = intrinsic[2];
            cy = intrinsic[3];

            scalar_t v_x, v_y, v_z, v2_x, v2_y, v2_z;
            v_x = (y - cx) * pixel / fx;
            v_y = (x - cy) * pixel / fy;
            v_z = pixel;
            
            if(debug)
                printf("V: %f %f %f \n", v_x, v_y, v_z);

            scalar_t v_norm = norm(-v_x, -v_y, -v_z);
            if(debug)
                printf("V_norm: %f \n", v_norm);

            if(v_norm <= 0.)
                return;
            v_norm = sqrt(v_norm);
            v2_x = -v_x / v_norm;
            v2_y = -v_y / v_norm;
            v2_z = -v_z / v_norm;
            if(debug) {
                printf("-V_normalized %f, %f, %f \n", v2_x, v2_y, v2_z);
                printf("SubMatrix1: \n");
            }

            scalar_t max_dot1, max_dot2, max_dot3, max_dot4;
            max_dot1 = -1.;
            max_dot2 = -1.;
            max_dot3 = -1.;
            max_dot4 = -1.;

             for(int i=-radius; i<=0; i++) {
                for(int j=-radius; j<=0; j++) {
                    if(x+i <0 || x+i >= height || y+j < 0 || y+j >= width)
                        break;
                    int temp_index = index + i*width + j;
                    scalar_t temp_pixel = in_depth[temp_index];
                    if (temp_pixel==0.)
                        continue;

                    scalar_t c_x, c_y, c_z;
                    c_x = (y+j - cx) * temp_pixel / fx;
                    c_y = (x+i - cy) * temp_pixel / fy;
                    c_z = temp_pixel;

                    c_x = c_x - v_x;
                    c_y = c_y - v_y;
                    c_z = c_z - v_z;
                    scalar_t c_norm = norm(c_x, c_y, c_z);
                    if(c_norm <= 0.)
                        continue;
                    c_norm = sqrt(c_norm);
                    c_x = c_x / c_norm;
                    c_y = c_y / c_norm;
                    c_z = c_z / c_norm;
                    scalar_t dot_prod = c_x*v2_x + c_y*v2_y + c_z*v2_z;
                    if(debug) {
                        printf("%d, %d : %f %f %f \n", i, j, c_x, c_y, c_z);
                        printf("DotProd: %f \n", dot_prod);
                    }
                    if(dot_prod > max_dot1) {
                        max_dot1 = dot_prod;
                    }

                 }
             }

             if(debug)
                printf("SubMatrix2: \n");
             for(int i=0; i<=radius; i++) {
                for(int j=-radius; j<=0; j++) {
                    if(x+i <0 || x+i >= height || y+j < 0 || y+j >= width)
                        break;
                    int temp_index = index + i*width + j;
                    scalar_t temp_pixel = in_depth[temp_index];
                    if (temp_pixel==0.)
                        continue;

                    scalar_t c_x, c_y, c_z;
                    c_x = (y+j - cx) * temp_pixel / fx;
                    c_y = (x+i - cy) * temp_pixel / fy;
                    c_z = temp_pixel;
                    c_x = c_x - v_x;
                    c_y = c_y - v_y;
                    c_z = c_z - v_z;
                    scalar_t c_norm = norm(c_x, c_y, c_z);
                    if(c_norm <= 0.)
                        continue;
                    c_norm = sqrt(c_norm);
                    c_x = c_x / c_norm;
                    c_y = c_y / c_norm;
                    c_z = c_z / c_norm;
                    scalar_t dot_prod = c_x*v2_x + c_y*v2_y + c_z*v2_z;
                    if(debug) {
                        printf("%d, %d : %f %f %f \n", i, j, c_x, c_y, c_z);
                        printf("DotProd: %f \n", dot_prod);
                    }
                    if(dot_prod > max_dot2) {
                        max_dot2 = dot_prod;
                    }

                 }
             }

             if(debug)
                printf("SubMatrix3: \n");
             for(int i=-radius; i<=0; i++) {
                for(int j=0; j<=radius; j++) {
                    if(x+i <0 || x+i >= height || y+j < 0 || y+j >= width)
                        break;
                    int temp_index = index + i*width + j;
                    scalar_t temp_pixel = in_depth[temp_index];
                    if (temp_pixel==0.)
                        continue;

                    scalar_t c_x, c_y, c_z;
                    c_x = (y+j - cx) * temp_pixel / fx;
                    c_y = (x+i - cy) * temp_pixel / fy;
                    c_z = temp_pixel;
                    c_x = c_x - v_x;
                    c_y = c_y - v_y;
                    c_z = c_z - v_z;
                    scalar_t c_norm = norm(c_x, c_y, c_z);
                    if(c_norm <= 0.)
                        continue;
                    c_norm = sqrt(c_norm);
                    c_x = c_x / c_norm;
                    c_y = c_y / c_norm;
                    c_z = c_z / c_norm;
                    scalar_t dot_prod = c_x*v2_x + c_y*v2_y + c_z*v2_z;
                    if(debug) {
                        printf("%d, %d : %f %f %f \n", i, j, c_x, c_y, c_z);
                        printf("DotProd: %f \n", dot_prod);
                    }
                    if(dot_prod > max_dot3) {
                        max_dot3 = dot_prod;
                    }

                 }
             }

             if(debug)
                printf("SubMatrix4: \n");
             for(int i=0; i<=radius; i++) {
                for(int j=0; j<=radius; j++) {
                    if(x+i <0 || x+i >= height || y+j < 0 || y+j >= width)
                        break;
                    int temp_index = index + i*width + j;
                    scalar_t temp_pixel = in_depth[temp_index];
                    if(temp_pixel==0.)
                        continue;

                    scalar_t c_x, c_y, c_z;
                    c_x = (y+j - cx) * temp_pixel / fx;
                    c_y = (x+i - cy) * temp_pixel / fy;
                    c_z = temp_pixel;
                    c_x = c_x - v_x;
                    c_y = c_y - v_y;
                    c_z = c_z - v_z;
                    scalar_t c_norm = norm(c_x, c_y, c_z);
                    if(c_norm <= 0.)
                        continue;
                    c_norm = sqrt(c_norm);
                    c_x = c_x / c_norm;
                    c_y = c_y / c_norm;
                    c_z = c_z / c_norm;
                    scalar_t dot_prod = c_x*v2_x + c_y*v2_y + c_z*v2_z;
                    if(debug) {
                        printf("%d, %d : %f %f %f \n", i, j, c_x, c_y, c_z);
                        printf("DotProd: %f \n", dot_prod);
                    }
                    if(dot_prod > max_dot4) {
                        max_dot4 = dot_prod;
                    }

                 }
             }

             if(max_dot1 + max_dot2 + max_dot3 + max_dot4 >= threshold) {
                  out[index] = 0.;
             }
        }

    }
}


template <typename scalar_t>
__global__ void downsample_flow_kernel(
    const scalar_t* __restrict__ in, 
    scalar_t* __restrict__  out, 
    unsigned int width_out, unsigned int height_out, unsigned int kernel) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // clamp to edge of image
    // x = max(0, x);
    //x = min(x, height_out-1);
    //y = max(y, 0);
    //y = min(y, width_out-1);

    scalar_t center = ((scalar_t)kernel-1.0)/2.0;
    
    unsigned int in_index = (kernel*x) * (kernel*width_out) + (kernel*y);
    unsigned int out_index = x*width_out + y;

    if (x >= 0 && x < height_out && y >= 0 && y < width_out) {
        //printf("out_index: %d, %d   -   in_pixel: %d, %d\n", x, y, x*kernel, y*kernel);
        scalar_t mean_u=0;
        scalar_t mean_v=0;
        scalar_t weights=0.0;
        unsigned int count = 0;
        for(int i=0; i<kernel; i++) {
            for(int j=0; j<kernel; j++) {
                int temp_index = in_index + (i*kernel)*width_out + j;
                scalar_t temp_u = in[2*temp_index];
                scalar_t temp_v = in[2*temp_index+1];
                if(temp_u != 0. || temp_v != 0.) {
                    scalar_t weight = max(0.0, 1.0-(abs(i-center)/kernel)) * max(0.0, 1.0-(abs(j-center)/kernel));
                    count += 1;
                    weights += weight;
                    mean_u += temp_u * weight;
                    mean_v += temp_v * weight;
                }
            }
        }
        if(count != 0) {
            mean_u = mean_u / weights;
            mean_v = mean_v / weights;
        }
        else {
            mean_u = 0.f;
            mean_v = 0.f;
        }
        out[2*out_index] = mean_u;
        out[2*out_index+1] = mean_v;
    }
}


__global__ void downsample_mask_kernel(
    const int* __restrict__ in, 
    int* __restrict__  out, 
    unsigned int width_out, unsigned int height_out, unsigned int kernel) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // clamp to edge of image
    // x = max(0, x);
    //x = min(x, height_out-1);
    //y = max(y, 0);
    //y = min(y, width_out-1);
    
    unsigned int in_index = (kernel*x) * (kernel*width_out) + (kernel*y);
    unsigned int out_index = x*width_out + y;

    if (x >= 0 && x < height_out && y >= 0 && y < width_out) {
        //printf("out_index: %d, %d   -   in_pixel: %d, %d\n", x, y, x*kernel, y*kernel);
        bool active = false;
        for(int i=0; i<kernel; i++) {
            for(int j=0; j<kernel; j++) {
                int temp_index = in_index + (i*kernel)*width_out + j;
                int temp_mask = in[temp_index];
                if(temp_mask != 0.) {
                    active = true;
                    break;
                }
            }
        }
        if(active) {
            out[out_index] = 1;
        }
        else {
            out[out_index] = 0;
        }
    }
}



at::Tensor depth_image_cuda(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output, unsigned int size, unsigned int width,unsigned int height) {
    //dim3 threads(BLOCK_W, BLOCK_H);
    //dim3 blocks((height)/TILE_W, (width)/TILE_H);
    dim3 threads(512);
    dim3 blocks(size/512+1);
    depth_image_kernel<<<blocks, threads>>>(input_uv.data_ptr<int>(), input_depth.data_ptr<float>(), output.data_ptr<float>(), size, width, height);
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}


at::Tensor visibility_filter_cuda(at::Tensor input, at::Tensor output, unsigned int width,unsigned int height, unsigned int threshold) {
    //dim3 threads(BLOCK_W, BLOCK_H);
    //dim3 blocks((height)/TILE_W, (width)/TILE_H);
    dim3 threads(32, 32);
    dim3 blocks(height/32, width/32);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "visibility_filter", ([&] {
        visibility_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), width, height, threshold);}));
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}

at::Tensor visibility_filter_cuda2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor output, unsigned int width,unsigned int height, float threshold, unsigned int radius) {
    //dim3 threads(BLOCK_W, BLOCK_H);
    //dim3 blocks((height)/TILE_W, (width)/TILE_H);
    dim3 threads(32, 32);
    dim3 blocks(height/32+1, width/32+1);
    AT_DISPATCH_FLOATING_TYPES(input_depth.scalar_type(), "visibility_filter2", ([&] {
        visibility_kernel2<scalar_t><<<blocks, threads>>>(input_depth.data_ptr<scalar_t>(), intrinsic.data_ptr<scalar_t>(),
                                                          output.data_ptr<scalar_t>(), width, height, threshold, radius);}));
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}

at::Tensor downsample_flow_cuda(at::Tensor input_uv, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    //dim3 threads(BLOCK_W, BLOCK_H);
    //dim3 blocks((height)/TILE_W, (width)/TILE_H);
    dim3 threads(32, 32);
    dim3 blocks(height_out/32+1, width_out/32+1);
    AT_DISPATCH_FLOATING_TYPES(input_uv.scalar_type(), "downsample_flow", ([&] {
        downsample_flow_kernel<scalar_t><<<blocks, threads>>>(input_uv.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                                              width_out, height_out, kernel);}));
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}

at::Tensor downsample_mask_cuda(at::Tensor input_mask, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    //dim3 threads(BLOCK_W, BLOCK_H);
    //dim3 blocks((height)/TILE_W, (width)/TILE_H);
    dim3 threads(32, 32);
    dim3 blocks(height_out/32+1, width_out/32+1);
    downsample_mask_kernel<<<blocks, threads>>>(input_mask.data_ptr<int>(), output.data_ptr<int>(),
                                              width_out, height_out, kernel);
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}

/*at::Tensor visibility_filter_cuda_shared(at::Tensor input_depth, at::Tensor input_points, at::Tensor output, unsigned int width,unsigned int height, float threshold) {
    dim3 threads(BLOCK_W, BLOCK_H);
    dim3 blocks((height)/TILE_W, (width)/TILE_H);
    //dim3 threads(32, 32);
    //dim3 blocks(height/32, width/32);
    AT_DISPATCH_FLOATING_TYPES(input_depth.type(), "visibility_filter2", ([&] {
        visibility_kernel_shared<scalar_t><<<blocks, threads>>>(input_depth.data<scalar_t>(), input_points.data<scalar_t>(), output.data<scalar_t>(), width, height, threshold);}));
    //visibility_kernel<float><<<blocks, threads>>>(input.data<float>(), output.data<float>(), width, height, threshold);
    return output;
}*/
