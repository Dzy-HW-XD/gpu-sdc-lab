/*
 * reduce.cu - deterministic FP32 block-reduction workload for the GPU SDC Lab.
 *
 * Each block sums a strided chunk of a deterministic input array, performs a
 * shared-memory tree reduction, then thread 0:
 *   - writes the block sum to `partials[blockIdx.x]`  (deterministic observable)
 *   - does `atomicAdd(&g_sum, block_sum)`             (ATOM.ADD.F32)
 *   - does `atomicAdd(&g_count, 1)`                   (ATOM.ADD.S32)
 *
 * The observable compared by the oracle is the `partials` array (bit-exact,
 * because per-block order is fixed). The float atomic accumulator is
 * order-dependent (not bit-exact) and is recorded in metadata only; the integer
 * atomic counter is deterministic. The returned value of an atomic is used (via
 * the `keep alive` branch) so nvcc emits ATOM (destination register) rather
 * than RED (no destination), making the atom instruction group injectable.
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

__global__ void reduce_blocks(const float *__restrict__ in, int N,
                              float *__restrict__ partials,
                              float *__restrict__ g_sum,
                              int *__restrict__ g_count,
                              float *__restrict__ g_dummy) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local = 0.0f;
    for (; i < N; i += stride) {
        local += in[i];
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float bs = sdata[0];
        partials[blockIdx.x] = bs;                 // deterministic observable
        float prev = atomicAdd(g_sum, bs);         // ATOM.ADD.F32 (dest = old)
        int oldc = atomicAdd(g_count, 1);          // ATOM.ADD.S32
        if (prev == 1234567.0f || oldc < 0) {      // keep returned values live
            *g_dummy = prev + (float)oldc;
        }
    }
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

static std::string arg_value(int argc, char **argv, const char *name,
                             const char *def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0)
            return std::string(argv[i + 1]);
    return std::string(def);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    int N = atoi(arg_value(argc, argv, "--N", "1048576").c_str());
    int blocks = atoi(arg_value(argc, argv, "--blocks", "256").c_str());
    int threads = atoi(arg_value(argc, argv, "--threads", "256").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string dtype = arg_value(argc, argv, "--dtype", "fp32");
    std::string out = arg_value(argc, argv, "--out", "reduce_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    if (dtype != "fp32") {
        fprintf(stdout, "ERROR: unsupported dtype '%s' (Phase 1 supports fp32 only)\n",
                dtype.c_str());
        return 3;
    }

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    float *hIn = (float *)malloc((size_t)N * sizeof(float));
    float *hPart = (float *)malloc((size_t)blocks * sizeof(float));
    if (!hIn || !hPart) {
        fprintf(stdout, "ERROR: host malloc failed\n");
        return 4;
    }
    std::string input = arg_value(argc, argv, "--input", "");
    if (!input.empty()) {
        if (!read_f32(input, hIn, (size_t)N)) return 6;
    } else {
        for (int i = 0; i < N; ++i) hIn[i] = det_value((u64)i, seed);
    }

    float *dIn = NULL, *dPart = NULL, *dSum = NULL, *dDummy = NULL;
    int *dCount = NULL;
    CUDA_CHECK(cudaMalloc(&dIn, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPart, (size_t)blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dDummy, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dSum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(dCount, 0, sizeof(int)));

    reduce_blocks<<<blocks, threads, threads * sizeof(float)>>>(
        dIn, N, dPart, dSum, dCount, dDummy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hPart, dPart, (size_t)blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float hSum = 0.0f, hDummy = 0.0f;
    int hCount = 0;
    CUDA_CHECK(cudaMemcpy(&hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hDummy, dDummy, sizeof(float), cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < blocks; ++i) checksum += (double)hPart[i];

    std::string binpath = out + ".bin";
    FILE *fp = fopen(binpath.c_str(), "wb");
    if (!fp) {
        fprintf(stdout, "ERROR: cannot open output '%s'\n", binpath.c_str());
        return 5;
    }
    fwrite(hPart, sizeof(float), blocks, fp);
    fclose(fp);

    fp = fopen((out + ".json").c_str(), "w");
    if (fp) {
        fprintf(fp,
                "{\"N\":%d,\"blocks\":%d,\"threads\":%d,\"dtype\":\"%s\","
                "\"seed\":%llu,\"device\":%d,\"device_name\":\"%s\","
                "\"checksum\":%.10g,\"atomic_sum\":%.10g,\"atomic_count\":%d}\n",
                N, blocks, threads, dtype.c_str(), seed, device, prop.name,
                checksum, (double)hSum, hCount);
        fclose(fp);
    }

    printf("REDUCE_DONE N=%d blocks=%d threads=%d dtype=%s seed=%llu checksum=%.10g\n",
           N, blocks, threads, dtype.c_str(), seed, checksum);

    free(hIn); free(hPart);
    cudaFree(dIn); cudaFree(dPart); cudaFree(dSum); cudaFree(dCount); cudaFree(dDummy);
    return 0;
}
