/*
 * gemm.cu - deterministic FP32 GEMM workload for the GPU SDC Lab.
 *
 * C = A * B   (row-major, A is MxK, B is KxN, C is MxN)
 *
 * Inputs are generated from a deterministic function of the element index and
 * a user-provided seed. No randomness, no time dependence: repeated runs on the
 * same binary/GPU produce bit-identical output. This makes it a valid golden
 * oracle for SDC comparison.
 *
 * Output:
 *   <out>.bin   raw little-endian float32 C matrix (M*N elements)
 *   <out>.json  metadata (M,N,K,dtype,seed,pattern,checksum,device)
 *
 * The program prints a short deterministic status line and exits non-zero on
 * any CUDA error so the RuntimeObserver can classify crashes.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stdout, "ERROR: CUDA error at %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(_e));                         \
            fflush(stdout);                                                    \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

typedef unsigned long long u64;

// Deterministic pseudo-random float in [-1, 1) from an index.
__host__ __device__ static inline float det_value(u64 idx, u64 seed) {
    u64 x = idx + 0x9E3779B97F4A7C15ULL * (seed + 1ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    double v = (double)((x >> 40) & 0xFFFFFFULL) / 8388608.0 - 1.0; // [-1,1)
    return (float)v;
}

// muladd == 0: default FP32 accumulation (compiles to FFMA) -> fp32 stream is
//              essentially all FFMA.
// muladd == 1: forced separate round-to-nearest multiply and add, so the fp32
//              instruction stream contains FMUL and FADD (for the
//              instruction-type study). Deterministic either way.
__global__ void gemm_fp32(const float *A, const float *B, float *C, int M, int N,
                          int K, int muladd) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        if (muladd) {
            for (int k = 0; k < K; ++k) {
                acc = __fadd_rn(__fmul_rn(A[(u64)row * K + k],
                                          B[(u64)k * N + col]), acc);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                acc += A[(u64)row * K + k] * B[(u64)k * N + col];
            }
        }
        C[(u64)row * N + col] = acc;
    }
}

static std::string arg_value(int argc, char **argv, const char *name,
                             const char *def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0)
            return std::string(argv[i + 1]);
    return std::string(def);
}

static bool read_f32(const std::string &path, float *dst, size_t n) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stdout, "ERROR: cannot open input '%s'\n", path.c_str());
        return false;
    }
    size_t got = fread(dst, sizeof(float), n, fp);
    fclose(fp);
    if (got != n) {
        fprintf(stdout, "ERROR: input '%s' has %zu < %zu floats\n", path.c_str(), got, n);
        return false;
    }
    return true;
}

static double pattern_scale(const std::string &p) {
    if (p == "small") return 1e-3;
    if (p == "large") return 1e3;
    if (p == "near_zero") return 1e-30;
    return 1.0;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    int M = atoi(arg_value(argc, argv, "--M", "1024").c_str());
    int N = atoi(arg_value(argc, argv, "--N", "1024").c_str());
    int K = atoi(arg_value(argc, argv, "--K", "1024").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string dtype = arg_value(argc, argv, "--dtype", "fp32");
    std::string pattern = arg_value(argc, argv, "--pattern", "normal");
    std::string out = arg_value(argc, argv, "--out", "gemm_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());
    std::string op_mode = arg_value(argc, argv, "--op-mode", "fma");
    int muladd = (op_mode == "muladd") ? 1 : 0;

    if (dtype != "fp32") {
        fprintf(stdout, "ERROR: unsupported dtype '%s' (Phase 1 supports fp32 only)\n",
                dtype.c_str());
        return 3;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t a_sz = (size_t)M * K, b_sz = (size_t)K * N, c_sz = (size_t)M * N;

    float *hA = (float *)malloc(a_sz * sizeof(float));
    float *hB = (float *)malloc(b_sz * sizeof(float));
    float *hC = (float *)malloc(c_sz * sizeof(float));
    if (!hA || !hB || !hC) {
        fprintf(stdout, "ERROR: host malloc failed\n");
        return 4;
    }

    std::string inputA = arg_value(argc, argv, "--inputA", "");
    std::string inputB = arg_value(argc, argv, "--inputB", "");
    double scale = pattern_scale(pattern);
    bool mixed = (pattern == "mixed");
    if (!inputA.empty()) {
        if (!read_f32(inputA, hA, a_sz)) return 6;
    } else {
        for (size_t i = 0; i < a_sz; ++i) {
            float v = det_value((u64)i, seed) * (float)scale;
            if (mixed) v *= (i % 3 == 0) ? 1e3f : ((i % 3 == 1) ? 1e-3f : 1.0f);
            hA[i] = v;
        }
    }
    if (!inputB.empty()) {
        if (!read_f32(inputB, hB, b_sz)) return 6;
    } else {
        for (size_t i = 0; i < b_sz; ++i) {
            float v = det_value((u64)i + 0x1000000ULL, seed + 7ULL) * (float)scale;
            if (mixed) v *= (i % 3 == 0) ? 1e3f : ((i % 3 == 1) ? 1e-3f : 1.0f);
            hB[i] = v;
        }
    }

    float *dA = NULL, *dB = NULL, *dC = NULL;
    CUDA_CHECK(cudaMalloc(&dA, a_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, b_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, c_sz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, a_sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, b_sz * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_fp32<<<grid, block>>>(dA, dB, dC, M, N, K, muladd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, c_sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Deterministic checksum (sequential host sum) for sanity/logging only.
    // An injected fault can make the output NaN/Inf; emit valid JSON in that
    // case (json.parse rejects lowercase `nan`).
    double csum = 0.0;
    bool csum_finite = true;
    for (size_t i = 0; i < c_sz; ++i) csum += (double)hC[i];
    if (!(csum == csum) || csum > 1e308 || csum < -1e308) csum_finite = false;
    char csum_json[64];
    if (csum_finite) snprintf(csum_json, sizeof(csum_json), "%.10g", csum);
    else snprintf(csum_json, sizeof(csum_json), "\"nan\"");

    std::string binpath = out + ".bin";
    FILE *fp = fopen(binpath.c_str(), "wb");
    if (!fp) {
        fprintf(stdout, "ERROR: cannot open output '%s'\n", binpath.c_str());
        return 5;
    }
    fwrite(hC, sizeof(float), c_sz, fp);
    fclose(fp);

    std::string jsonpath = out + ".json";
    fp = fopen(jsonpath.c_str(), "w");
    if (fp) {
        fprintf(fp,
                "{\"M\":%d,\"N\":%d,\"K\":%d,\"dtype\":\"%s\",\"seed\":%llu,"
                "\"pattern\":\"%s\",\"op_mode\":\"%s\",\"device\":%d,\"device_name\":\"%s\","
                "\"elements\":%llu,\"checksum\":%s}\n",
                M, N, K, dtype.c_str(), seed, pattern.c_str(), op_mode.c_str(), device,
                prop.name, (unsigned long long)c_sz, csum_json);
        fclose(fp);
    }

    printf("GEMM_DONE M=%d N=%d K=%d dtype=%s seed=%llu pattern=%s op_mode=%s checksum=%.10g\n",
           M, N, K, dtype.c_str(), seed, pattern.c_str(), op_mode.c_str(), csum);

    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
