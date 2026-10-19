/*
 * attention_forward.cu - causal self-attention forward pass for SDC studies.
 *
 *   X [S,D] (bf16)  ->  Q=X Wq, K=X Wk, V=X Wv          (cuBLAS, BF16 in / FP32 out)
 *   per head h:  Sh = Qh Kh^T / sqrt(dh)                 (cuBLAS, per-head launch)
 *                Ph = softmax_causal(Sh)                 (custom FP32 kernel)
 *                Oh = Ph Vh                              (cuBLAS, per-head launch)
 *   Ocat = concat(Oh)  ->  Y = Ocat Wo                   (cuBLAS)
 *
 * Design choices:
 *  - The two projection GEMMs and the score/PV GEMMs are separate kernel
 *    launches; score and PV are launched once PER HEAD so that a fault can be
 *    attributed to a specific head by selecting the kernel invocation index
 *    (cuBLAS batched GEMM would hide the head mapping inside one kernel).
 *  - BF16 x BF16 tensor-core GEMM with FP32 accumulation (CUBLAS_COMPUTE_32F),
 *    atomics disabled for bit-reproducible golden runs.
 *  - Q/K/V (FP32 projection outputs) are rounded to BF16 before the next GEMM,
 *    matching a BF16 attention data path. Softmax is explicit FP32.
 *  - Intermediate tensors can be dumped (raw little-endian float32) for
 *    propagation analysis.
 *
 * Outputs (prefix = --out):
 *   <out>.Y.bin     [S,D]      final output (always)
 *   <out>.Q/K/V.bin [S,D]      projections            (--dump 1)
 *   <out>.S.bin     [H,S,S]    raw scores QhKh^T      (--dump 1)
 *   <out>.P.bin     [H,S,S]    softmax probabilities  (--dump 1)
 *   <out>.O.bin     [H,S,dh]   per-head outputs       (--dump 1)
 *   <out>.Ocat.bin  [S,D]      concatenated heads     (--dump 1)
 *   <out>.json                 metadata + checksum
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>

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

#define CUBLAS_CHECK(call)                                                      \
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

static void write_f32(const std::string &path, const float *h, size_t n) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stdout, "ERROR: cannot open output '%s'\n", path.c_str());
        exit(5);
    }
    if (n) fwrite(h, sizeof(float), n, fp);
    fclose(fp);
}

// ---------------------------------------------------------------------------
__global__ void cast_f32_bf16(const float *in, __nv_bfloat16 *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}

// One block per (head, query row). Causal mask + softmax in FP32.
__global__ void softmax_causal(const float *Sc, float *P, __nv_bfloat16 *Pb,
                               int S, int dh) {
    extern __shared__ float red[];
    int bid = blockIdx.x;      // h*S + i
    int i = bid % S;
    const float *row = Sc + (size_t)bid * S;
    float *prow = P + (size_t)bid * S;
    __nv_bfloat16 *pbrow = Pb + (size_t)bid * S;
    int tid = threadIdx.x, nt = blockDim.x;
    float scale = 1.0f / sqrtf((float)dh);

    float m = -INFINITY;
    for (int j = tid; j <= i; j += nt) m = fmaxf(m, row[j] * scale);
    red[tid] = m;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j <= i; j += nt) {
        float e = expf(row[j] * scale - m);
        prow[j] = e;
        sum += e;
    }
    for (int j = tid; j < S; j += nt)
        if (j > i) prow[j] = 0.0f;
    red[tid] = sum;
    __syncthreads();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int j = tid; j <= i; j += nt) {
        float p = prow[j] * inv;
        prow[j] = p;
        pbrow[j] = __float2bfloat16_rn(p);
    }
    for (int j = tid; j < S; j += nt)
        if (j > i) pbrow[j] = __float2bfloat16_rn(0.0f);
}

// Ocat[i, h*dh+j] = O[h,i,j]
__global__ void concat_heads(const float *O, float *Ocat, long total,
                             int S, int D, int H, int dh) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = (int)(idx % dh);
    int h = (int)((idx / dh) % H);
    int i = (int)(idx / ((long)dh * H));
    Ocat[(size_t)i * D + h * dh + j] = O[(size_t)(h * S + i) * dh + j];
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    int S = atoi(arg_value(argc, argv, "--S", "128").c_str());
    int D = atoi(arg_value(argc, argv, "--D", "2048").c_str());
    int H = atoi(arg_value(argc, argv, "--H", "1").c_str());
    int dump = atoi(arg_value(argc, argv, "--dump", "1").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string dtype = arg_value(argc, argv, "--dtype", "bf16");
    std::string out = arg_value(argc, argv, "--out", "attention_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    if (dtype != "bf16" && dtype != "bfloat16") {
        fprintf(stdout, "ERROR: unsupported dtype '%s' (bf16 only)\n", dtype.c_str());
        return 3;
    }
    if (S <= 0 || D <= 0 || H <= 0 || (D % H) != 0) {
        fprintf(stdout, "ERROR: invalid shapes S=%d D=%d H=%d (need D%%H==0)\n",
                S, D, H);
        return 3;
    }
    int dh = D / H;

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t x_sz = (size_t)S * D;          // bf16
    size_t w_sz = (size_t)D * D;          // bf16 per weight
    size_t proj_sz = (size_t)S * D;       // fp32
    size_t score_sz = (size_t)H * S * S;  // fp32
    size_t oh_sz = (size_t)H * S * dh;    // fp32

    __nv_bfloat16 *hX = (__nv_bfloat16 *)malloc(x_sz * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hWq = (__nv_bfloat16 *)malloc(w_sz * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hWk = (__nv_bfloat16 *)malloc(w_sz * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hWv = (__nv_bfloat16 *)malloc(w_sz * sizeof(__nv_bfloat16));
    __nv_bfloat16 *hWo = (__nv_bfloat16 *)malloc(w_sz * sizeof(__nv_bfloat16));
    float *hY = (float *)malloc(proj_sz * sizeof(float));
    if (!hX || !hWq || !hWk || !hWv || !hWo || !hY) {
        fprintf(stdout, "ERROR: host malloc failed\n");
        return 4;
    }

    std::string inputX = arg_value(argc, argv, "--inputX", "");
    if (!inputX.empty()) {
        if (!read_bf16(inputX, hX, x_sz)) return 6;
    } else {
        for (size_t i = 0; i < x_sz; ++i)
            hX[i] = __float2bfloat16_rn(det_value((u64)i, seed));
    }
    for (size_t i = 0; i < w_sz; ++i) {
        hWq[i] = __float2bfloat16_rn(det_value((u64)i + 0x2000000ULL, seed + 11ULL));
        hWk[i] = __float2bfloat16_rn(det_value((u64)i + 0x4000000ULL, seed + 13ULL));
        hWv[i] = __float2bfloat16_rn(det_value((u64)i + 0x6000000ULL, seed + 17ULL));
        hWo[i] = __float2bfloat16_rn(det_value((u64)i + 0x8000000ULL, seed + 19ULL));
    }

    __nv_bfloat16 *dX = NULL, *dWq = NULL, *dWk = NULL, *dWv = NULL, *dWo = NULL;
    float *dQ = NULL, *dK = NULL, *dV = NULL;
    __nv_bfloat16 *dQb = NULL, *dKb = NULL, *dVb = NULL;
    float *dSc = NULL, *dP = NULL, *dOh = NULL, *dOcat = NULL, *dY = NULL;
    __nv_bfloat16 *dPb = NULL, *dOcatb = NULL;

    CUDA_CHECK(cudaMalloc(&dX, x_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dWq, w_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dWk, w_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dWv, w_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dWo, w_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dQ, proj_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, proj_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, proj_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQb, proj_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dKb, proj_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dVb, proj_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dSc, score_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, score_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPb, score_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dOh, oh_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOcat, proj_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOcatb, proj_sz * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dY, proj_sz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX, hX, x_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq, w_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWk, hWk, w_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWv, hWv, w_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWo, hWo, w_sz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    const float alpha = 1.0f, beta = 0.0f;
    const __nv_bfloat16 *bf = dX; (void)bf;

    // Q = X Wq (row-major, C[0]=A[0]*B[0] via the (N,M,K) mapping)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S, D,
                              &alpha, dWq, CUDA_R_16BF, D, dX, CUDA_R_16BF, D,
                              &beta, dQ, CUDA_R_32F, D,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S, D,
                              &alpha, dWk, CUDA_R_16BF, D, dX, CUDA_R_16BF, D,
                              &beta, dK, CUDA_R_32F, D,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S, D,
                              &alpha, dWv, CUDA_R_16BF, D, dX, CUDA_R_16BF, D,
                              &beta, dV, CUDA_R_32F, D,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    long proj_n = (long)proj_sz;
    int cast_threads = 256;
    cast_f32_bf16<<<(proj_n + cast_threads - 1) / cast_threads, cast_threads>>>(
        dQ, dQb, proj_n);
    cast_f32_bf16<<<(proj_n + cast_threads - 1) / cast_threads, cast_threads>>>(
        dK, dKb, proj_n);
    cast_f32_bf16<<<(proj_n + cast_threads - 1) / cast_threads, cast_threads>>>(
        dV, dVb, proj_n);
    CUDA_CHECK(cudaGetLastError());

    // Sh = Qh Kh^T : row-major MxN = A(MxK) * B(NxK)^T  ->  gemm(OP_T, OP_N)
    for (int h = 0; h < H; ++h) {
        size_t off = (size_t)h * dh;
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, S, S, dh, &alpha,
            dKb + off, CUDA_R_16BF, D,
            dQb + off, CUDA_R_16BF, D,
            &beta, dSc + (size_t)h * S * S, CUDA_R_32F, S,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // softmax over each head's rows (causal)
    int sm_threads = 256;
    size_t sm_shmem = (size_t)sm_threads * sizeof(float);
    softmax_causal<<<H * S, sm_threads, sm_shmem>>>(dSc, dP, dPb, S, dh);
    CUDA_CHECK(cudaGetLastError());

    // Oh = Ph Vh : row-major (MxN) = P(MxK) * V(KxN)  -> gemm(OP_N, OP_N)
    for (int h = 0; h < H; ++h) {
        size_t off = (size_t)h * dh;
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, dh, S, S, &alpha,
            dVb + off, CUDA_R_16BF, D,
            dPb + (size_t)h * S * S, CUDA_R_16BF, S,
            &beta, dOh + (size_t)h * S * dh, CUDA_R_32F, dh,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    long cat_total = (long)S * D;
    concat_heads<<<(cat_total + cast_threads - 1) / cast_threads, cast_threads>>>(
        dOh, dOcat, cat_total, S, D, H, dh);
    cast_f32_bf16<<<(cat_total + cast_threads - 1) / cast_threads, cast_threads>>>(
        dOcat, dOcatb, cat_total);
    CUDA_CHECK(cudaGetLastError());

    // Y = Ocat Wo
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S, D,
                              &alpha, dWo, CUDA_R_16BF, D, dOcatb, CUDA_R_16BF, D,
                              &beta, dY, CUDA_R_32F, D,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hY, dY, proj_sz * sizeof(float), cudaMemcpyDeviceToHost));

    double csum = 0.0;
    bool finite = true;
    for (size_t i = 0; i < proj_sz; ++i) {
        csum += (double)hY[i];
        if (!(hY[i] == hY[i]) || hY[i] == INFINITY || hY[i] == -INFINITY)
            finite = false;
    }

    write_f32(out + ".Y.bin", hY, proj_sz);

    if (dump) {
        float *hbuf = (float *)malloc(score_sz * sizeof(float));
        float *href = (float *)malloc(proj_sz * sizeof(float));
        if (!hbuf || !href) { fprintf(stdout, "ERROR: dump malloc failed\n"); return 4; }
        CUDA_CHECK(cudaMemcpy(href, dQ, proj_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".Q.bin", href, proj_sz);
        CUDA_CHECK(cudaMemcpy(href, dK, proj_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".K.bin", href, proj_sz);
        CUDA_CHECK(cudaMemcpy(href, dV, proj_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".V.bin", href, proj_sz);
        CUDA_CHECK(cudaMemcpy(hbuf, dSc, score_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".S.bin", hbuf, score_sz);
        CUDA_CHECK(cudaMemcpy(hbuf, dP, score_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".P.bin", hbuf, score_sz);
        float *hoh = (float *)malloc(oh_sz * sizeof(float));
        if (hoh) {
            CUDA_CHECK(cudaMemcpy(hoh, dOh, oh_sz * sizeof(float), cudaMemcpyDeviceToHost));
            write_f32(out + ".O.bin", hoh, oh_sz);
            free(hoh);
        }
        CUDA_CHECK(cudaMemcpy(href, dOcat, proj_sz * sizeof(float), cudaMemcpyDeviceToHost));
        write_f32(out + ".Ocat.bin", href, proj_sz);
        free(hbuf);
        free(href);
    }

    FILE *fp = fopen((out + ".json").c_str(), "w");
    if (fp) {
        fprintf(fp,
                "{\"S\":%d,\"D\":%d,\"H\":%d,\"dh\":%d,\"dtype\":\"%s\","
                "\"accumulate\":\"fp32\",\"mask\":\"causal\",\"backend\":\"cublas\","
                "\"seed\":%llu,\"device\":%d,\"device_name\":\"%s\","
                "\"elements\":%llu,\"dump\":%d,\"finite\":%s,\"checksum\":%.10g}\n",
                S, D, H, dh, dtype.c_str(), seed, device, prop.name,
                (unsigned long long)proj_sz, dump,
                finite ? "true" : "false", csum);
        fclose(fp);
    }

    printf("ATTENTION_DONE S=%d D=%d H=%d dh=%d seed=%llu finite=%d checksum=%.10g\n",
           S, D, H, dh, seed, (int)finite, csum);

    free(hX); free(hWq); free(hWk); free(hWv); free(hWo); free(hY);
    cudaFree(dX); cudaFree(dWq); cudaFree(dWk); cudaFree(dWv); cudaFree(dWo);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dQb); cudaFree(dKb); cudaFree(dVb);
    cudaFree(dSc); cudaFree(dP); cudaFree(dPb);
    cudaFree(dOh); cudaFree(dOcat); cudaFree(dOcatb); cudaFree(dY);
    cublasDestroy(handle);
    return 0;
}
