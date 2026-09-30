/*
 * attention.cu - deterministic single-head FP32 attention (explicit math path,
 * no flash/SDPA).  X [S, 3D] packs Q,K,V.  O = softmax(Q K^T / sqrt(D)) V.
 * One block per query row; fixed-order reductions -> bit-exact golden.
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

__global__ void attn_kernel(const float *X, float *O, int S, int D) {
    int i = blockIdx.x;
    if (i >= S) return;
    extern __shared__ float sm[];
    float *w = sm;             // S weights/scores
    float *red = sm + S;       // blockDim.x reduction scratch
    int tid = threadIdx.x, nt = blockDim.x;
    float rsqrt = 1.0f / sqrtf((float)D);
    const float *Xi = X + (size_t)i * 3 * D;

    float lmax = -INFINITY;
    for (int j = tid; j < S; j += nt) {
        const float *Kj = X + (size_t)j * 3 * D + D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += Xi[d] * Kj[d];
        float sc = dot * rsqrt;
        w[j] = sc;
        lmax = fmaxf(lmax, sc);
    }
    red[tid] = lmax; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid+st]); __syncthreads(); }
    float m = red[0]; __syncthreads();

    float lsum = 0.0f;
    for (int j = tid; j < S; j += nt) { w[j] = expf(w[j] - m); lsum += w[j]; }
    red[tid] = lsum; __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid+st]; __syncthreads(); }
    float inv = 1.0f / red[0]; __syncthreads();
    for (int j = tid; j < S; j += nt) w[j] *= inv;

    for (int d = tid; d < D; d += nt) {
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) acc += w[j] * X[(size_t)j * 3 * D + 2 * D + d];
        O[(size_t)i * D + d] = acc;
    }
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
    int S = atoi(arg_value(argc, argv, "--S", "128").c_str());
    int D = atoi(arg_value(argc, argv, "--D", "64").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string out = arg_value(argc, argv, "--out", "attention_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    CUDA_CHECK(cudaSetDevice(device));
    size_t x_sz = (size_t)S * 3 * D, o_sz = (size_t)S * D;
    float *hX = (float*)malloc(x_sz*sizeof(float)), *hO = (float*)malloc(o_sz*sizeof(float));
    if (!hX || !hO) { fprintf(stdout, "ERROR: host malloc failed\n"); return 4; }

    std::string input = arg_value(argc, argv, "--input", "");
    if (!input.empty()) { if (!read_f32(input, hX, x_sz)) return 6; }
    else for (size_t i = 0; i < x_sz; ++i) hX[i] = det_value(i, seed);

    float *dX=NULL,*dO=NULL;
    CUDA_CHECK(cudaMalloc(&dX, x_sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, o_sz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX, x_sz*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 128;
    size_t shmem = (size_t)(S + threads) * sizeof(float);
    attn_kernel<<<S, threads, shmem>>>(dX, dO, S, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO, dO, o_sz*sizeof(float), cudaMemcpyDeviceToHost));

    double csum = 0.0; for (size_t i = 0; i < o_sz; ++i) csum += hO[i];

    FILE *fp = fopen((out + ".bin").c_str(), "wb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open output\n"); return 5; }
    fwrite(hO, sizeof(float), o_sz, fp); fclose(fp);
    fp = fopen((out + ".json").c_str(), "w");
    if (fp) { fprintf(fp, "{\"S\":%d,\"D\":%d,\"seed\":%llu,\"checksum\":%.10g}\n", S, D, seed, csum); fclose(fp); }
    printf("ATTENTION_DONE S=%d D=%d seed=%llu checksum=%.10g\n", S, D, seed, csum);

    free(hX); free(hO); cudaFree(dX); cudaFree(dO);
    return 0;
}
