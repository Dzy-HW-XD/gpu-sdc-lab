// qkv_v CUTLASS kernel: V[M,N] = A[M,K] * W[N,K]^T
// A/W are raw BF16 little-endian; output V is raw BF16.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>

#ifndef VARIANT
#define VARIANT 1
#endif

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

#if VARIANT == 1
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int Stages = 3;
#elif VARIANT == 2
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int Stages = 5;
#elif VARIANT == 3
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int Stages = 3;
#elif VARIANT == 4
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int Stages = 3;
#else
#error "bad VARIANT"
#endif

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 1, ElementAccumulator, ElementAccumulator>;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp, ThreadblockSwizzle, Stages, 8, 8>;

static std::string arg(int argc, char **argv, const char *k, const char *d) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == k) return argv[i + 1];
    return d;
}

static void die(const char *msg) { fprintf(stderr, "%s\n", msg); exit(2); }

int main(int argc, char **argv) {
    const int M = std::atoi(arg(argc, argv, "--M", "2264").c_str());
    const int N = std::atoi(arg(argc, argv, "--N", "256").c_str());
    const int K = std::atoi(arg(argc, argv, "--K", "1536").c_str());
    const std::string a_path = arg(argc, argv, "--inputA", "A.bin");
    const std::string b_path = arg(argc, argv, "--inputB", "W.bin");
    const std::string out = arg(argc, argv, "--out", "V.bin");
    const int dev = std::atoi(arg(argc, argv, "--device", "0").c_str());
    cudaSetDevice(dev);

    size_t as = (size_t)M * K, bs = (size_t)N * K, cs = (size_t)M * N;
    std::vector<__nv_bfloat16> hA(as), hB(bs), hC(cs);
    FILE *f = fopen(a_path.c_str(), "rb");
    if (!f) die("open A failed");
    if (fread(hA.data(), sizeof(__nv_bfloat16), as, f) != as) die("read A short");
    fclose(f);
    f = fopen(b_path.c_str(), "rb");
    if (!f) die("open W failed");
    if (fread(hB.data(), sizeof(__nv_bfloat16), bs, f) != bs) die("read W short");
    fclose(f);

    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, as * sizeof(__nv_bfloat16));
    cudaMalloc(&dB, bs * sizeof(__nv_bfloat16));
    cudaMalloc(&dC, cs * sizeof(__nv_bfloat16));
    cudaMemcpy(dA, hA.data(), as * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bs * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, cs * sizeof(__nv_bfloat16));

    Gemm gemm;
    cutlass::gemm::GemmCoord problem(M, N, K);
    ElementAccumulator alpha = 1.0f, beta = 0.0f;
    using TensorRefA = cutlass::TensorRef<ElementA const, LayoutA>;
    using TensorRefB = cutlass::TensorRef<ElementB const, LayoutB>;
    using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
    typename Gemm::Arguments args(
        problem,
        TensorRefA(reinterpret_cast<ElementA const *>(dA), LayoutA(K)),
        TensorRefB(reinterpret_cast<ElementB const *>(dB), LayoutB(K)),
        TensorRefC(reinterpret_cast<ElementC *>(dC), LayoutC(N)),
        TensorRefC(reinterpret_cast<ElementC *>(dC), LayoutC(N)),
        typename EpilogueOp::Params(alpha, beta));
    cutlass::Status status = gemm(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS status %d\n", (int)status);
        return 3;
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e));
        return 4;
    }
    cudaMemcpy(hC.data(), dC, cs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    f = fopen(out.c_str(), "wb");
    if (!f) die("open out failed");
    fwrite(hC.data(), sizeof(__nv_bfloat16), cs, f);
    fclose(f);
    double sum = 0;
    for (size_t i = 0; i < cs; ++i) sum += __bfloat162float(hC[i]);
    printf("QKV_V_CUTLASS_DONE variant=%d M=%d N=%d K=%d checksum=%.10g\n",
           VARIANT, M, N, K, sum);
    return 0;
}
