# Paper abstract (draft)

**Working title**

> **Input Matters: Characterizing and Exploiting Input-Dependent Silent Data
> Corruption in GPU Kernels**

---

## Abstract (English)

Silent data corruption (SDC) — a computation that completes without any error
signal but returns a wrong result — is a first-order reliability threat for GPUs
in datacenters. Prior GPU fault-injection studies largely treat the *fault site*
(instruction, register, bit) as the object of study and hold the *data* fixed,
implicitly assuming that a fault's propagation is a property of the hardware
alone. We argue and show that, for the same fault site, whether an error remains
**masked**, becomes **silent data corruption**, or escalates to a **crash** is
strongly governed by the *input distribution* of the kernel.

We present a controlled, reproducible measurement methodology and an
open-source, extensible fault-injection laboratory. The key design is a *paired*
experiment: a single dynamic instruction/register/bit is selected with
NVBitFI-based dynamic binary instrumentation, and only the input tensor is
varied, so that the observed outcome is attributable to the data. Using a single
NVIDIA A100, we characterize six representative kernel families — GEMM,
reduction/atomics, 2-D convolution, softmax, attention, and a small end-to-end
training step — under eight synthetic input families (e.g. wide-dynamic-range,
sparse, cancellation-prone, near-overflow) plus a real data tensor.

We contribute: (i) evidence that SDC probability and error magnitude vary by
orders of magnitude with input distribution, and are jointly modulated by
instruction class and by FP32 bit field (sign/exponent/mantissa); (ii)
cross-workload regularities and a multi-layer propagation study for a training
step; and (iii) a lightweight input-feature model that predicts SDC risk and
drives an *input-aware* injection/sampling policy, increasing the number of SDCs
discovered per injection relative to uniform sampling. Our implementation and
all experiment configurations are released as a reproducible artifact.

**Keywords:** silent data corruption, GPU reliability, fault injection, NVBitFI,
input-dependent error propagation, SDC testing.

---

## 摘要（中文）

静默数据错误（SDC）——计算没有任何报错信号却返回错误结果——已成为数据中心
GPU 的一阶可靠性威胁。以往的 GPU 注错研究多以“故障点（指令/寄存器/位）”为
研究对象，而把“数据”固定不变，隐含假设故障传播是硬件自身的属性。我们提出
并证明：对同一个故障点，一个错误最终是被**掩盖（masked）**、变成**静默数据
错误（SDC）**、还是升级为**崩溃（crash）**，强烈地由该 kernel 的**输入分布**
决定。

我们给出了一套受控、可复现的测量方法学，以及一个开源、可扩展的故障注入实验
台。核心设计是**配对实验**：用基于 NVBitFI 的动态二进制插桩选定同一条动态
指令/寄存器/位，只改变输入张量，从而把观测到的结局干净地归因于数据。在单张
NVIDIA A100 上，我们刻画了六类代表性 kernel——GEMM、归约/原子、2D 卷积、
softmax、attention、以及一个小型端到端训练步——在八类合成输入（如大动态范围、
稀疏、灾难性抵消、接近溢出）加一个真实数据张量下的行为。

我们的贡献包括：（i）SDC 概率与误差幅度随输入分布变化可达数量级，并受指令
类别与 FP32 位区（符号/指数/尾数）共同调制；（ii）跨 workload 的规律性，以及
训练步的多层传播研究；（iii）一个轻量的输入特征模型，用于预测 SDC 风险并驱动
**输入感知（input-aware）** 的注错/采样策略，使“每次注错发现的 SDC 数”相对
均匀采样显著提升。实现与全部实验配置以可复现 artifact 形式公开。

**关键词：** 静默数据错误、GPU 可靠性、故障注入、NVBitFI、输入相关的错误传播、
SDC 测试。
