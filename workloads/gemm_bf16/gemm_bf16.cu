/*
 * gemm_bf16.cu - BF16 x BF16 Tensor Core GEMM with FP32 accumulation (cuBLAS).
 *
 *   C = A * B   (row-major, A is MxK bf16, B is KxN bf16, C is MxN fp32)
 *
 * The matrix multiply is executed by cuBLAS (cublasGemmEx) using BF16 inputs
 * and FP32 accumulation (CUBLAS_COMPUTE_32F), so on sm_80 the inner loop is
 * HMMA (Tensor Core) with FP32 accumulator registers -- this is the operator
 * we inject faults into.
 *
 * Determinism: atomics are disabled (cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_
 * ALLOWED)) so no split-K atomics are used; combined with a fixed algorithm
 * the same inputs/gpu produce bit-identical output. (Verified by the lab.)
 *
 * Inputs (bf16, raw little-endian uint16) can be supplied with --inputA/--inputB;
 * otherwise a deterministic bf16 pattern is generated from the seed.
 *
 * Output: <out>.bin (raw little-endian float32 C) + <out>.json metadata.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stdout, "ERROR: cuBLAS error %d at %s:%d\n", (int)_s,      \
                    __FILE__, __LINE__);                                       \
            fflush(stdout);                                                    \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

typedef unsigned long long u64;

// Deterministic pseudo-random float in [-1, 1) from an index (same as gemm.cu).
__host__ __device__ static inline float det_value(u64 idx, u64 seed) {
    u64 x = idx + 0x9E3779B97F4A7C15ULL * (seed + 1ULL);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    double v = (double)((x >> 40) & 0xFFFFFFULL) / 8388608.0 - 1.0;
    return (float)v;
}

static std::string arg_value(int argc, char **argv, const char *name,
                             const char *def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0)
            return std::string(argv[i + 1]);
    return std::string(def);
}

// Read n bf16 elements (raw uint16) into a host buffer.
static bool read_bf16(const std::string &path, __nv_bfloat16 *dst, size_t n) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stdout, "ERROR: cannot open input '%s'\n", path.c_str());
        return false;
    }
    size_t got = fread(dst, sizeof(__nv_bfloat16), n, fp);
    fclose(fp);
    if (got != n) {
        fprintf(stdout, "ERROR: input '%s' has %zu < %zu bf16 elements\n",
                path.c_str(), got, n);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    int M = atoi(arg_value(argc, argv, "--M", "1024").c_str());
    int N = atoi(arg_value(argc, argv, "--N", "1024").c_str());
    int K = atoi(arg_value(argc, argv, "--K", "1024").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string dtype = arg_value(argc, argv, "--dtype", "bf16");
    std::string out = arg_value(argc, argv, "--out", "gemm_bf16_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    if (dtype != "bf16" && dtype != "bfloat16") {
        fprintf(stdout, "ERROR: unsupported dtype '%s' (bf16 only)\n", dtype.c_str());
        return 3;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t a_sz = (size_t)M * K, b_sz = (size_t)K * N, c_sz = (size_t)M * N;

    __nv_bfloat16 *hA = (__nv_bfloat16 *)malloc(a_sz * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hB = (__nv_bfloat16 *)malloc(b_sz * sizeof(__nv_bfloat16));
    float *hC = (float *)malloc(c_sz * sizeof(float));
    if (!hA || !hB || !hC) { fprintf(stdout, "ERROR: host malloc failed\n"); return 4; }

    std::string inputA = arg_value(argc, argv, "--inputA", "");
    std::string inputB = arg_value(argc, argv, "--inputB", "");
    if (!inputA.empty()) {
        if (!read_bf16(inputA, hA, a_sz)) return 6;
    } else {
        for (size_t i = 0; i < a_sz; ++i)
            hA[i] = __float2bfloat16_rn(det_value((u64)i, seed));
    }
    if (!inputB.empty()) {
        if (!read_bf16(inputB, hB, b_sz)) return 6;
    } else {
        for (size_t i = 0; i < b_sz; ++i)
            hB[i] = __float2bfloat16_rn(det_value((u64)i + 0x1000000ULL, seed + 7ULL));
    }

    __nv_bfloat16 *dA = NULL, *dB = NULL;
    float *dC = NULL;
    CUDA_CHECK(cudaMalloc(&dA, a_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB, b_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dC, c_sz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, a_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, b_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // deterministic: no split-K atomics
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));

    // Row-major C(MxN) = A(MxK)*B(KxN) computed as column-major:
    //   C_cm(NxM) = B^T * A^T  =>  gemm(OP_N, OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        dB, CUDA_R_16BF, N,
        dA, CUDA_R_16BF, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, c_sz * sizeof(float), cudaMemcpyDeviceToHost));

    double csum = 0.0;
    bool csum_finite = true;
    for (size_t i = 0; i < c_sz; ++i) csum += (double)hC[i];
    if (!(csum == csum) || csum > 1e308 || csum < -1e308) csum_finite = false;
    char csum_json[64];
    if (csum_finite) snprintf(csum_json, sizeof(csum_json), "%.10g", csum);
    else snprintf(csum_json, sizeof(csum_json), "\"nan\"");

    std::string binpath = out + ".bin";
    FILE *fp = fopen(binpath.c_str(), "wb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open output '%s'\n", binpath.c_str()); return 5; }
    fwrite(hC, sizeof(float), c_sz, fp);
    fclose(fp);

    std::string jsonpath = out + ".json";
    fp = fopen(jsonpath.c_str(), "w");
    if (fp) {
        fprintf(fp,
                "{\"M\":%d,\"N\":%d,\"K\":%d,\"dtype\":\"%s\",\"seed\":%llu,"
                "\"device\":%d,\"device_name\":\"%s\",\"elements\":%llu,"
                "\"accumulate\":\"fp32\",\"backend\":\"cublas\",\"checksum\":%s}\n",
                M, N, K, dtype.c_str(), seed, device, prop.name,
                (unsigned long long)c_sz, csum_json);
        fclose(fp);
    }

    printf("GEMM_BF16_DONE M=%d N=%d K=%d dtype=%s seed=%llu checksum=%.10g\n",
           M, N, K, dtype.c_str(), seed, csum);

    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    return 0;
}
