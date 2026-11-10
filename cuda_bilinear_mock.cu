// ============================================================
// nvcc -O3 -arch=sm_86 -o cuda_bilinear_mock cuda_bilinear_mock.cu
// ============================================================
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================
// CUDA Kernel: 双线性插值
// ============================================================
__global__ void bilinear_interp_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale_h, float scale_w
) {
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (dst_y >= dst_h || dst_x >= dst_w) return;

    // 映射回源图坐标（float）
    float src_y = (float)dst_y / scale_h;
    float src_x = (float)dst_x / scale_w;

    int y0 = floorf(src_y);
    int x0 = floorf(src_x);
    int y1 = min(y0 + 1, src_h - 1);
    int x1 = min(x0 + 1, src_w - 1);

    float dy = src_y - y0;
    float dx = src_x - x0;

    // 4 个邻近点
    float v00 = src[y0 * src_w + x0];
    float v01 = src[y0 * src_w + x1];
    float v10 = src[y1 * src_w + x0];
    float v11 = src[y1 * src_w + x1];

    // 双线性插值: 先水平再垂直
    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    float val = v0 * (1.0f - dy) + v1 * dy;

    dst[dst_y * dst_w + dst_x] = val;
}

// ============================================================
// CPU 参考实现
// ============================================================
void bilinear_interp_cpu(const float* src, float* dst,
                         int src_h, int src_w, float scale) {
    int dst_h = (int)(src_h * scale);
    int dst_w = (int)(src_w * scale);

    for (int dy = 0; dy < dst_h; dy++) {
        float sy = (float)dy / scale;
        int y0 = (int)floorf(sy);
        int y1 = min(y0 + 1, src_h - 1);
        float fy = sy - y0;

        for (int dx = 0; dx < dst_w; dx++) {
            float sx = (float)dx / scale;
            int x0 = (int)floorf(sx);
            int x1 = min(x0 + 1, src_w - 1);
            float fx = sx - x0;

            float v00 = src[y0 * src_w + x0];
            float v01 = src[y0 * src_w + x1];
            float v10 = src[y1 * src_w + x0];
            float v11 = src[y1 * src_w + x1];

            float v0 = v00 * (1-fx) + v01 * fx;
            float v1 = v10 * (1-fx) + v11 * fx;
            dst[dy * dst_w + dx] = v0 * (1-fy) + v1 * fy;
        }
    }
}

// ============================================================
// 主函数
// ============================================================
int main() {
    // ---- 参数 ----
    int src_h = 4, src_w = 4;
    float scale = 2.0f;
    int dst_h = (int)(src_h * scale);
    int dst_w = (int)(src_w * scale);

    size_t src_bytes = src_h * src_w * sizeof(float);
    size_t dst_bytes = dst_h * dst_w * sizeof(float);

    printf("源图: %d x %d\n", src_h, src_w);
    printf("目标: %d x %d (scale=%.1f)\n\n", dst_h, dst_w, scale);

    // ---- 分配 host 内存 ----
    float* h_src = (float*)malloc(src_bytes);
    float* h_dst_gpu = (float*)malloc(dst_bytes);
    float* h_dst_cpu = (float*)malloc(dst_bytes);

    // 填充源图（简单的梯度图，方便肉眼验证）
    for (int i = 0; i < src_h * src_w; i++) {
        h_src[i] = (float)i;
    }

    printf("源图数据:\n");
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++)
            printf("%5.1f ", h_src[y * src_w + x]);
        printf("\n");
    }
    printf("\n");

    // ---- 分配 device 内存 ----
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_bytes, cudaMemcpyHostToDevice));

    // ---- 启动 kernel ----
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bilinear_interp_kernel<<<grid, block>>>(
        d_src, d_dst, src_h, src_w, dst_h, dst_w, scale, scale
    );
    cudaEventRecord(stop);
    CUDA_CHECK(cudaGetLastError());

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // ---- 拷回结果 ----
    CUDA_CHECK(cudaMemcpy(h_dst_gpu, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    // ---- CPU 参考 ----
    bilinear_interp_cpu(h_src, h_dst_cpu, src_h, src_w, scale);

    // ---- 对比 ----
    float max_err = 0;
    for (int i = 0; i < dst_h * dst_w; i++) {
        float err = fabsf(h_dst_gpu[i] - h_dst_cpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("GPU 结果:\n");
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++)
            printf("%6.2f ", h_dst_gpu[y * dst_w + x]);
        printf("\n");
    }
    printf("\n");
    printf("GPU 耗时: %.3f ms\n", ms);
    printf("最大误差: %e\n", max_err);
    printf("验证: %s\n", max_err < 1e-4 ? " PASS" : " FAIL");

    // ---- 清理 ----
    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    free(h_dst_gpu);
    free(h_dst_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}