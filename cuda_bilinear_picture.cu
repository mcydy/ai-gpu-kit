// ============================================================
// nvcc -o cuda_bilinear_picture cuda_bilinear_picture.cu -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
// ============================================================
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
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
// CUDA Kernel: RGB 双线性插值
// ============================================================
__global__ void bilinear_rgb_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale_h, float scale_w
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float src_x = ((float)dst_x + 0.5f) / scale_w - 0.5f;
    float src_y = ((float)dst_y + 0.5f) / scale_h - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1 < src_w ? x0 + 1 : x0;
    int y1 = y0 + 1 < src_h ? y0 + 1 : y0;

    float dx = src_x - x0;
    float dy = src_y - y0;

    int src_stride = src_w * 3;
    int dst_stride = dst_w * 3;

    for (int c = 0; c < 3; c++) {
        float v00 = (float)src[y0 * src_stride + x0 * 3 + c];
        float v01 = (float)src[y0 * src_stride + x1 * 3 + c];
        float v10 = (float)src[y1 * src_stride + x0 * 3 + c];
        float v11 = (float)src[y1 * src_stride + x1 * 3 + c];

        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        float val = v0 * (1.0f - dy) + v1 * dy;

        int out = (int)(val + 0.5f);
        out = out < 0 ? 0 : (out > 255 ? 255 : out);
        dst[dst_y * dst_stride + dst_x * 3 + c] = (unsigned char)out;
    }
}

// ============================================================
// CPU 参考实现
// ============================================================
static void bilinear_rgb_cpu(const unsigned char* src, unsigned char* dst,
                              int src_h, int src_w, float scale) {
    int dst_h = (int)(src_h * scale);
    int dst_w = (int)(src_w * scale);

    for (int dy = 0; dy < dst_h; dy++) {
        float sy = ((float)dy + 0.5f) / scale - 0.5f;
        sy = fmaxf(0, fminf(sy, src_h - 1));
        int y0 = (int)floorf(sy), y1 = y0 + 1 < src_h ? y0 + 1 : y0;
        float fy = sy - y0;

        for (int dx = 0; dx < dst_w; dx++) {
            float sx = ((float)dx + 0.5f) / scale - 0.5f;
            sx = fmaxf(0, fminf(sx, src_w - 1));
            int x0 = (int)floorf(sx), x1 = x0 + 1 < src_w ? x0 + 1 : x0;
            float fx = sx - x0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[y0*src_w*3 + x0 * 3 + c];
                float v01 = src[y0*src_w*3 + x1 * 3 + c];
                float v10 = src[y1*src_w*3 + x0 * 3 + c];
                float v11 = src[y1*src_w*3 + x1 * 3 + c];

                float v0 = v00*(1-fx) + v01*fx;
                float v1 = v10*(1-fx) + v11*fx;
                float val = v0*(1-fy) + v1*fy;

                int out = (int)(val + 0.5f);
                out = out < 0 ? 0 : (out > 255 ? 255 : out);
                dst[dy*dst_w*3 + dx*3 + c] = (unsigned char)out;
            }
        }
    }
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char** argv) {
    const char* input_path  = (argc > 1) ? argv[1] : "input.png";
    const char* output_path = (argc > 2) ? argv[2] : "output_gpu.jpg";
    float scale = (argc > 3) ? atof(argv[3]) : 0.5f;

    // ---- 1. OpenCV 读图 ----
    cv::Mat img_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        fprintf(stderr, " 无法加载图片: %s\n", input_path);
        return 1;
    }

    // BGR → RGB
    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    int src_h = img_rgb.rows;
    int src_w = img_rgb.cols;
    int dst_h = (int)(src_h * scale);
    int dst_w = (int)(src_w * scale);

    size_t src_bytes = src_h * src_w * 3 * sizeof(unsigned char);
    size_t dst_bytes = dst_h * dst_w * 3 * sizeof(unsigned char);

    printf("┌──────────────────────────────────────────┐\n");
    printf("│ 输入: %-36s│\n", input_path);
    printf("│ 源图: %d x %d x 3                        │\n", src_w, src_h);
    printf("│ 目标: %d x %d (scale=%.1f)                │\n", dst_w, dst_h, scale);
    printf("└──────────────────────────────────────────┘\n\n");

    // ---- 2. 分配 host 内存 ----
    unsigned char* h_dst_gpu = (unsigned char*)malloc(dst_bytes);
    unsigned char* h_dst_cpu = (unsigned char*)malloc(dst_bytes);

    // ---- 3. GPU 显存分配 + H→D 拷贝 ----
    unsigned char *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));
    CUDA_CHECK(cudaMemcpy(d_src, img_rgb.data, src_bytes, cudaMemcpyHostToDevice));

    // ---- 4. 启动 kernel ----
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bilinear_rgb_kernel<<<grid, block>>>(
        d_src, d_dst, src_h, src_w, dst_h, dst_w, scale, scale
    );
    cudaEventRecord(stop);
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // ---- 5. D→H 拷回 ----
    CUDA_CHECK(cudaMemcpy(h_dst_gpu, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    // ---- 6. CPU 参考 ----
    auto t0 = std::chrono::high_resolution_clock::now();
    bilinear_rgb_cpu(img_rgb.data, h_dst_cpu, src_h, src_w, scale);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 7. 对比 ----
    long long max_diff = 0, total_diff = 0;
    for (size_t i = 0; i < dst_h * dst_w * 3; i++) {
        long long d = abs((long long)h_dst_gpu[i] - (long long)h_dst_cpu[i]);
        if (d > max_diff) max_diff = d;
        total_diff += d;
    }

    // ---- 8. OpenCV 写回 ----
    cv::Mat out_gpu(dst_h, dst_w, CV_8UC3, h_dst_gpu);
    cv::Mat out_gpu_bgr;
    cv::cvtColor(out_gpu, out_gpu_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, out_gpu_bgr);

    cv::Mat out_cpu(dst_h, dst_w, CV_8UC3, h_dst_cpu);
    cv::Mat out_cpu_bgr;
    cv::cvtColor(out_cpu, out_cpu_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite("output_cpu.jpg", out_cpu_bgr);

    // ---- 9. 打印 ----
    printf("GPU 耗时:   %8.3f ms\n", gpu_ms);
    printf("CPU 耗时:   %8.3f ms\n", cpu_ms);
    printf("加速比:     %8.2fx\n", cpu_ms / gpu_ms);
    printf("最大像素差: %8lld  (允许 ≤2)\n", max_diff);
    printf("平均像素差: %8.4f\n", (double)total_diff / (dst_h * dst_w * 3));
    printf("验证:       %s\n", max_diff <= 2 ? "✅ PASS" : "⚠️  minor diff");
    printf("\n输出:\n  GPU: %s\n  CPU: output_cpu.jpg\n", output_path);

    // ---- 10. 清理 ----
    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_dst_gpu);
    free(h_dst_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}