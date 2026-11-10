#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 第一步 从 HBM读取数据到 共享内存
    sdata[tid] = (idx < n) ? input[idx]:0.0f;
    __syncthreads();

    // 第二步 树形相加
    for (int s = blockDim.x /2; s > 0;   s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[s+tid];
        __syncthreads();
    }

    if ( 0 == tid)
        output[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 10, BLOCK = 4;                   // 写死，BLOCK 是 2 的幂
    const int GRID = (n + BLOCK - 1) / BLOCK;      // = 3，第三块只有 2 个有效元素

    float hIn[n], hOut[GRID];
    for (int i = 0; i < n; ++i) hIn[i] = (float)(i + 1);   // 1 2 3 ... 10

    printf("input: ");
    for (int i = 0; i < n; ++i) printf("%.0f ", hIn[i]);
    printf("\n");

    float *dIn, *dOut;
    cudaMalloc((void**)&dIn,  n     * sizeof(float));
    cudaMalloc((void**)&dOut, GRID  * sizeof(float));   // ★ 长度是 GRID，不是 n
    cudaMemcpy(dIn, hIn, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_kernel<<<GRID, BLOCK, BLOCK * sizeof(float)>>>(dIn, dOut, n);// 第三个参数给 extern __shared__ 分配大小，漏了就写野内存
    cudaMemcpy(hOut, dOut, GRID * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0;
    for (int b = 0; b < GRID; ++b) {
        printf("block %d 局部和 = %.0f\n", b, hOut[b]);
        total += hOut[b];                              // ★ 多 block 结果再 reduce 一次
    }
    printf("全局和 = %.0f  (CPU golden = %.0f)\n", total, (float)n * (n + 1) / 2);

    cudaFree(dIn); cudaFree(dOut);
    return 0;
}