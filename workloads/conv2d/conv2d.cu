/*
 * conv2d.cu - deterministic FP32 direct 2D convolution (NCHW, N=1).
 *
 * X [C_in,H,W], W [C_out,C_in,K,K], Y [C_out,H,W], stride=1, pad=K/2.
 * Deterministic (fixed accumulation order). Optional --input/--weight load raw
 * float32 files; otherwise inputs come from det_value().
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

__global__ void conv2d_kernel(const float *X, const float *W, float *Y,
                              int Cin, int Cout, int H, int Wd, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * H * Wd;
    if (idx >= total) return;
    int w = idx % Wd;
    int h = (idx / Wd) % H;
    int co = idx / (Wd * H);
    int pad = K / 2;
    float acc = 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int ih = h + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                int iw = w + kw - pad;
                if (iw < 0 || iw >= Wd) continue;
                acc += X[(ci * H + ih) * Wd + iw] *
                       W[((co * Cin + ci) * K + kh) * K + kw];
            }
        }
    }
    Y[idx] = acc;
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
    int Cin = atoi(arg_value(argc, argv, "--Cin", "8").c_str());
    int Cout = atoi(arg_value(argc, argv, "--Cout", "16").c_str());
    int H = atoi(arg_value(argc, argv, "--H", "32").c_str());
    int Wd = atoi(arg_value(argc, argv, "--W", "32").c_str());
    int K = atoi(arg_value(argc, argv, "--kernel", "3").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string out = arg_value(argc, argv, "--out", "conv2d_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    CUDA_CHECK(cudaSetDevice(device));
    size_t x_sz = (size_t)Cin * H * Wd, w_sz = (size_t)Cout * Cin * K * K, y_sz = (size_t)Cout * H * Wd;
    float *hX = (float*)malloc(x_sz*sizeof(float)), *hW = (float*)malloc(w_sz*sizeof(float)), *hY = (float*)malloc(y_sz*sizeof(float));
    if (!hX || !hW || !hY) { fprintf(stdout, "ERROR: host malloc failed\n"); return 4; }

    std::string input = arg_value(argc, argv, "--input", "");
    std::string weight = arg_value(argc, argv, "--weight", "");
    if (!input.empty()) { if (!read_f32(input, hX, x_sz)) return 6; }
    else for (size_t i = 0; i < x_sz; ++i) hX[i] = det_value(i, seed);
    if (!weight.empty()) { if (!read_f32(weight, hW, w_sz)) return 6; }
    else for (size_t i = 0; i < w_sz; ++i) hW[i] = det_value(i + 0x2000000ULL, seed + 3ULL) * 0.1f;

    float *dX=NULL,*dW=NULL,*dY=NULL;
    CUDA_CHECK(cudaMalloc(&dX, x_sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, w_sz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, y_sz*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, hX, x_sz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW, w_sz*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (int)((y_sz + threads - 1) / threads);
    conv2d_kernel<<<blocks, threads>>>(dX, dW, dY, Cin, Cout, H, Wd, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hY, dY, y_sz*sizeof(float), cudaMemcpyDeviceToHost));

    double csum = 0.0; for (size_t i = 0; i < y_sz; ++i) csum += hY[i];

    FILE *fp = fopen((out + ".bin").c_str(), "wb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open output\n"); return 5; }
    fwrite(hY, sizeof(float), y_sz, fp); fclose(fp);
    fp = fopen((out + ".json").c_str(), "w");
    if (fp) { fprintf(fp, "{\"Cin\":%d,\"Cout\":%d,\"H\":%d,\"W\":%d,\"kernel\":%d,\"seed\":%llu,\"checksum\":%.10g}\n",
                      Cin, Cout, H, Wd, K, seed, csum); fclose(fp); }
    printf("CONV2D_DONE Cin=%d Cout=%d H=%d W=%d K=%d seed=%llu checksum=%.10g\n", Cin, Cout, H, Wd, K, seed, csum);

    free(hX); free(hW); free(hY);
    cudaFree(dX); cudaFree(dW); cudaFree(dY);
    return 0;
}
