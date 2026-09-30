/*
 * mlp.cu - deterministic 2-layer MLP forward + backward (one training step).
 *
 *   X[B,D]  -> Z1 = X W1 + b1 -> A1 = ReLU(Z1)
 *   A1[H]   -> Z2 = A1 W2 + b2 -> P = softmax(Z2)
 *   loss    = -mean_b log P[b, y_b]
 *   backward: dZ2, dW2, dZ1, dW1  (SGD gradients)
 *
 * All reductions are fixed-order -> bit-exact golden. Emits several named
 * tensors so the oracle can observe error propagation at every layer:
 *   _a1, _logits, _probs, _loss, _dW1, _dW2
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

__global__ void lin1(const float *X, const float *W1, const float *b1,
                     float *Z1, float *A1, int B, int D, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, h = idx % H;
    float acc = b1[h];
    for (int d = 0; d < D; ++d) acc += X[b * D + d] * W1[d * H + h];
    Z1[idx] = acc;
    A1[idx] = acc > 0.0f ? acc : 0.0f;
}

__global__ void lin2(const float *A1, const float *W2, const float *b2,
                     float *Z2, int B, int H, int O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * O) return;
    int b = idx / O, o = idx % O;
    float acc = b2[o];
    for (int h = 0; h < H; ++h) acc += A1[b * H + h] * W2[h * O + o];
    Z2[idx] = acc;
}

__global__ void softmax_rows(const float *Z2, float *P, int B, int O) {
    int b = blockIdx.x;
    if (b >= B || threadIdx.x != 0) return;
    const float *z = Z2 + (size_t)b * O;
    float *p = P + (size_t)b * O;
    float m = -INFINITY;
    for (int o = 0; o < O; ++o) m = fmaxf(m, z[o]);
    float s = 0.0f;
    for (int o = 0; o < O; ++o) { p[o] = expf(z[o] - m); s += p[o]; }
    float inv = 1.0f / s;
    for (int o = 0; o < O; ++o) p[o] *= inv;
}

__global__ void loss_kernel(const float *P, const int *y, float *L, int B, int O) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    L[b] = -logf(P[(size_t)b * O + y[b]] + 1e-12f);
}

__global__ void dz2_kernel(const float *P, const int *y, float *dZ2, int B, int O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * O) return;
    int b = idx / O, o = idx % O;
    float t = (o == y[b]) ? 1.0f : 0.0f;
    dZ2[idx] = (P[idx] - t) / (float)B;
}

__global__ void dw2_kernel(const float *A1, const float *dZ2, float *dW2,
                           int B, int H, int O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * O) return;
    int h = idx / O, o = idx % O;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) acc += A1[b * H + h] * dZ2[b * O + o];
    dW2[idx] = acc;
}

__global__ void dz1_kernel(const float *dZ2, const float *W2, const float *Z1,
                           float *dZ1, int B, int H, int O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int b = idx / H, h = idx % H;
    float acc = 0.0f;
    for (int o = 0; o < O; ++o) acc += dZ2[b * O + o] * W2[h * O + o];
    dZ1[idx] = acc * (Z1[idx] > 0.0f ? 1.0f : 0.0f);
}

__global__ void dw1_kernel(const float *X, const float *dZ1, float *dW1,
                           int B, int D, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D * H) return;
    int d = idx / H, h = idx % H;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) acc += X[b * D + d] * dZ1[b * H + h];
    dW1[idx] = acc;
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
static void write_f32(const std::string &p, const float *src, size_t n) {
    FILE *fp = fopen(p.c_str(), "wb");
    if (!fp) { fprintf(stdout, "ERROR: cannot open '%s'\n", p.c_str()); exit(5); }
    fwrite(src, sizeof(float), n, fp); fclose(fp);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int B = atoi(arg_value(argc, argv, "--B", "64").c_str());
    int D = atoi(arg_value(argc, argv, "--D", "64").c_str());
    int H = atoi(arg_value(argc, argv, "--H", "128").c_str());
    int O = atoi(arg_value(argc, argv, "--O", "10").c_str());
    u64 seed = strtoull(arg_value(argc, argv, "--seed", "1234").c_str(), NULL, 10);
    std::string out = arg_value(argc, argv, "--out", "mlp_out");
    int device = atoi(arg_value(argc, argv, "--device", "0").c_str());

    CUDA_CHECK(cudaSetDevice(device));
    size_t x_sz = (size_t)B * D, w1_sz = (size_t)D * H, w2_sz = (size_t)H * O;
    size_t z1_sz = (size_t)B * H, a1_sz = z1_sz, z2_sz = (size_t)B * O, p_sz = z2_sz;

    float *hX=(float*)malloc(x_sz*4),*hW1=(float*)malloc(w1_sz*4),*hW2=(float*)malloc(w2_sz*4);
    float *hb1=(float*)malloc(H*4),*hb2=(float*)malloc(O*4);
    float *hZ1=(float*)malloc(z1_sz*4),*hA1=(float*)malloc(a1_sz*4),*hZ2=(float*)malloc(z2_sz*4),*hP=(float*)malloc(p_sz*4);
    float *hL=(float*)malloc(B*4),*hdW1=(float*)malloc(w1_sz*4),*hdW2=(float*)malloc(w2_sz*4);
    int *hy=(int*)malloc(B*sizeof(int));
    if (!hX||!hW1||!hW2||!hb1||!hb2||!hZ1||!hA1||!hZ2||!hP||!hL||!hdW1||!hdW2||!hy) {
        fprintf(stdout, "ERROR: host malloc failed\n"); return 4;
    }

    std::string input = arg_value(argc, argv, "--input", "");
    std::string wt1 = arg_value(argc, argv, "--weight1", "");
    std::string wt2 = arg_value(argc, argv, "--weight2", "");
    if (!input.empty()) { if (!read_f32(input, hX, x_sz)) return 6; }
    else for (size_t i = 0; i < x_sz; ++i) hX[i] = det_value(i, seed);
    if (!wt1.empty()) { if (!read_f32(wt1, hW1, w1_sz)) return 6; }
    else for (size_t i = 0; i < w1_sz; ++i) hW1[i] = det_value(i + 0x3000000ULL, seed + 11ULL) * 0.1f;
    if (!wt2.empty()) { if (!read_f32(wt2, hW2, w2_sz)) return 6; }
    else for (size_t i = 0; i < w2_sz; ++i) hW2[i] = det_value(i + 0x4000000ULL, seed + 13ULL) * 0.1f;
    for (int h = 0; h < H; ++h) hb1[h] = det_value(h + 0x5000000ULL, seed + 17ULL) * 0.01f;
    for (int o = 0; o < O; ++o) hb2[o] = det_value(o + 0x6000000ULL, seed + 19ULL) * 0.01f;
    for (int b = 0; b < B; ++b) hy[b] = (b * 7 + 3) % O;

    float *dX,*dW1,*dW2,*db1,*db2,*dZ1,*dA1,*dZ2,*dP,*dL,*ddW1,*ddW2; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, x_sz*4)); CUDA_CHECK(cudaMalloc(&dW1, w1_sz*4));
    CUDA_CHECK(cudaMalloc(&dW2, w2_sz*4)); CUDA_CHECK(cudaMalloc(&db1, H*4));
    CUDA_CHECK(cudaMalloc(&db2, O*4)); CUDA_CHECK(cudaMalloc(&dZ1, z1_sz*4));
    CUDA_CHECK(cudaMalloc(&dA1, a1_sz*4)); CUDA_CHECK(cudaMalloc(&dZ2, z2_sz*4));
    CUDA_CHECK(cudaMalloc(&dP, p_sz*4)); CUDA_CHECK(cudaMalloc(&dL, B*4));
    CUDA_CHECK(cudaMalloc(&ddW1, w1_sz*4)); CUDA_CHECK(cudaMalloc(&ddW2, w2_sz*4));
    CUDA_CHECK(cudaMalloc(&dy, B*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX, hX, x_sz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, hW1, w1_sz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, hW2, w2_sz*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, hb1, H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, hb2, O*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, B*sizeof(int), cudaMemcpyHostToDevice));

    int t = 256;
    lin1<<<(int)((z1_sz+t-1)/t), t>>>(dX, dW1, db1, dZ1, dA1, B, D, H);
    lin2<<<(int)((z2_sz+t-1)/t), t>>>(dA1, dW2, db2, dZ2, B, H, O);
    softmax_rows<<<B, 1>>>(dZ2, dP, B, O);
    loss_kernel<<<(int)((B+t-1)/t), t>>>(dP, dy, dL, B, O);
    dz2_kernel<<<(int)((z2_sz+t-1)/t), t>>>(dP, dy, dZ2, B, O);
    dw2_kernel<<<(int)((w2_sz+t-1)/t), t>>>(dA1, dZ2, ddW2, B, H, O);
    dz1_kernel<<<(int)((z1_sz+t-1)/t), t>>>(dZ2, dW2, dZ1, dZ1, B, H, O);
    dw1_kernel<<<(int)((w1_sz+t-1)/t), t>>>(dX, dZ1, ddW1, B, D, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // backward overwrote dZ2; re-run the forward pass to capture clean
    // A1 / logits(Z2) / probs(P) / loss for the oracle.
    lin1<<<(int)((z1_sz+t-1)/t), t>>>(dX, dW1, db1, dZ1, dA1, B, D, H);
    lin2<<<(int)((z2_sz+t-1)/t), t>>>(dA1, dW2, db2, dZ2, B, H, O);
    softmax_rows<<<B, 1>>>(dZ2, dP, B, O);
    loss_kernel<<<(int)((B+t-1)/t), t>>>(dP, dy, dL, B, O);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hA1, dA1, a1_sz*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hZ2, dZ2, z2_sz*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hP, dP, p_sz*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL, dL, B*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdW1, ddW1, w1_sz*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdW2, ddW2, w2_sz*4, cudaMemcpyDeviceToHost));

    double loss = 0.0; for (int b = 0; b < B; ++b) loss += hL[b];
    loss /= (double)B;
    double csum = 0.0; for (size_t i = 0; i < p_sz; ++i) csum += hP[i];

    write_f32(out + "_a1.bin", hA1, a1_sz);
    write_f32(out + "_logits.bin", hZ2, z2_sz);
    write_f32(out + "_probs.bin", hP, p_sz);
    write_f32(out + "_loss.bin", hL, B);
    write_f32(out + "_dW1.bin", hdW1, w1_sz);
    write_f32(out + "_dW2.bin", hdW2, w2_sz);
    FILE *fp = fopen((out + ".json").c_str(), "w");
    if (fp) { fprintf(fp, "{\"B\":%d,\"D\":%d,\"H\":%d,\"O\":%d,\"seed\":%llu,\"loss\":%.10g,\"checksum\":%.10g}\n",
                      B, D, H, O, seed, loss, csum); fclose(fp); }
    printf("MLP_DONE B=%d D=%d H=%d O=%d seed=%llu loss=%.10g checksum=%.10g\n",
           B, D, H, O, seed, loss, csum);

    return 0;
}
