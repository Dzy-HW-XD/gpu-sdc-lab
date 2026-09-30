/*
 * softmax.cu - deterministic FP32 row-wise softmax.  Y[r,:] = softmax(X[r,:]).
 * One block per row; fixed-order shared-memory reductions (deterministic).
 * Exercises fp32, ld, gp and MUFU/others (exp).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stdout,"ERROR: CUDA error at %s:%d: %s\n",__FILE__,__LINE__, \
            cudaGetErrorString(_e)); fflush(stdout); exit(2);} } while(0)

typedef unsigned long long u64;
__host__ __device__ static inline float det_value(u64 idx, u64 seed) {
    u64 x = idx + 0x9E3779B97F4A7C15ULL * (seed + 1ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27;
    x *= 0x94D049BB133111EBULL; x ^= x >> 31;
    return (float)((double)((x >> 40) & 0xFFFFFFULL) / 8388608.0 - 1.0);
}

__global__ void softmax_kernel(const float *X, float *Y, int R, int C) {
    int row = blockIdx.x;
    if (row >= R) return;
    extern __shared__ float sdata[];
    const float *xr = X + (size_t)row * C;
    float *yr = Y + (size_t)row * C;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    float m = -INFINITY;
    for (int i = tid; i < C; i += nt) m = fmaxf(m, xr[i]);
    sdata[tid] = m; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) sdata[tid] = fmaxf(sdata[tid], sdata[tid + st]);
        __syncthreads();
    }
    m = sdata[0]; __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < C; i += nt) {
        float e = expf(xr[i] - m);
        yr[i] = e;
        sum += e;
    }
    sdata[tid] = sum; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) sdata[tid] += sdata[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int i = tid; i < C; i += nt) yr[i] *= inv;
}

static std::string arg_value(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; ++i) if (strcmp(argv[i], name) == 0) return std::string(argv[i+1]);
    return std::string(def);
}
static bool read_f32(const std::string &p, float *dst, size_t n) {
    FILE *fp = fopen(p.c_str(), "rb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open '%s'\n", p.c_str()); return false; }
    size_t got = fread(dst, sizeof(float), n, fp); fclose(fp);
    if (got != n) { fprintf(stdout, "ERROR: '%s' has %zu < %zu\n", p.c_str(), got, n); return false; }
    return true;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int R = atoi(arg_value(argc, argv, "--R", "1024").c_str());
    int C = atoi(arg_value(argc, argv, "--C", "1024").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string out = arg_value(argc, argv, "--out", "softmax_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    CUDA_CHECK(cudaSetDevice(device));
    size_t n = (size_t)R * C;
    float *hX = (float*)malloc(n*sizeof(float)), *hY = (float*)malloc(n*sizeof(float));
    if (!hX || !hY) { fprintf(stdout, "ERROR: host malloc failed\n"); return 4; }

    std::string input = arg_value(argc, argv, "--input", "");
    if (!input.empty()) { if (!read_f32(input, hX, n)) return 6; }
    else for (size_t i = 0; i < n; ++i) hX[i] = det_value(i, seed) * 4.0f;

    float *dX=NULL,*dY=NULL;
    CUDA_CHECK(cudaMalloc(&dX, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX, n*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    softmax_kernel<<<R, threads, threads*sizeof(float)>>>(dX, dY, R, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hY, dY, n*sizeof(float), cudaMemcpyDeviceToHost));

    double csum = 0.0; for (size_t i = 0; i < n; ++i) csum += hY[i];

    FILE *fp = fopen((out + ".bin").c_str(), "wb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open output\n"); return 5; }
    fwrite(hY, sizeof(float), n, fp); fclose(fp);
    fp = fopen((out + ".json").c_str(), "w");
    if (fp) { fprintf(fp, "{\"R\":%d,\"C\":%d,\"seed\":%llu,\"checksum\":%.10g}\n", R, C, seed, csum); fclose(fp); }
    printf("SOFTMAX_DONE R=%d C=%d seed=%llu checksum=%.10g\n", R, C, seed, csum);

    free(hX); free(hY); cudaFree(dX); cudaFree(dY);
    return 0;
}
