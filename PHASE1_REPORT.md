# gpu-sdc-lab — Phase 1 report

Date: 2026-09-30 · Host `node20` · GPU index 1

## 1. Environment

| item | value |
|---|---|
| Host | node20, Ubuntu 24.04, kernel 6.8.0-142 |
| GPU | NVIDIA A100-SXM4-80GB (sm_80), physical index **1** |
| GPU UUID | GPU-f71ecf48-034d-ceba-f3f5-f7dfe4d1ec44 |
| Driver | **580.178.04** (CUDA 13.0) |
| GPU1 occupancy | shared with `VLLM::Worker_TP1` (~25 GiB); ~55 GiB free |
| Toolkits present | /usr/local/cuda-11.8, -13.0 (used), -13.3 |
| nvcc | 13.0.88 |
| gcc | 12.4.0 |
| Tool instrumentation | NVBit **v1.8.1** (Linux x86_64) |
| Fault injector | NVBitFI (`NVlabs/nvbitfi`), patched for A100+ CUDA 13.0 |
| Python | 3.12 (stdlib only; no numpy/pandas needed) |

NVBitFI compatibility patches (applied by `scripts/setup_nvbitfi.sh`, no system
changes):
1. profiler `inject_funcs.cu`: `ballot()` -> `__activemask()` / `__ballot_sync`
   (removed for sm_70+).
2. injector `inject_funcs.cu`: device-side `printf()/assert()` compiled out
   (they require CUDA >= 13.1; on 13.0 they raise `CUDA_ERROR_INVALID_SOURCE`).
Tools built with `ARCH=80`.

## 2. GEMM golden determinism

`experiments/001_gemm_baseline`, 10 independent runs of the 1024^3 FP32 GEMM:

- bitwise-identical runs: **10 / 10**
- `max_abs_error = 0.0`, `relative_l2_error = 0.0`

=> the golden oracle is **bit-reproducible**, so exact comparison
(`abs_tol = rel_tol = 0`) is a valid SDC threshold (no hand-tuned 1% cutoff).

## 3. NVBitFI injection status

Working. Verified end-to-end:
- profiler produced per-kernel group instruction counts;
- injector landed single faults (e.g. `mask=0x100`, `regNo=6`, `FADD@0x20`,
  `tid=3204` on the simple_add smoke test);
- 436 fault injections across 5 campaigns, **0 NOT_INJECTED** (every requested
  dynamic site executed);
- both clean exits with corrupted output and hard crashes were observed.

## 4. Fault-position definition

A fault site is
`(kernel_name, kernel_invocation_index, instruction_group, dynamic_inst_id)`.

- `dynamic_inst_id` = ordinal of the instruction **within the selected group**,
  counted per thread over the whole kernel invocation (per-group totals from a
  one-time `profiler.so` run of the exact workload).
- The bit is chosen with `bitIDSeed`; `bit = int(32*seed)` (single) so
  `seed=(bit+0.5)/32` targets an exact FP32 bit (31 sign, 23-30 exponent,
  0-22 mantissa).
- Groups used: `fp32` (FFMA/FADD), `ld` (LDG loads), `gp` (GPR-writing,
  including address arithmetic -> crashes).
- Experiment 002 buckets the group's dynamic range into 5 position bins
  (P0..P4) and samples 20 injections per bin. Every record also stores the
  resolved **static site** (`opcode@pcOffset`), register, thread id and
  mask/before/after values.

## 5. Supported fault types

| type | NVBitFI model id | meaning |
|---|---|---|
| `bitflip` | 0 | single bit flip in a destination register |
| `twobit` | 1 | two adjacent bits flipped |
| `random` | 2 | random value written |
| `zero` | 3 | register zeroed |

Instruction groups: `fp64`(0), `fp32`(1), `ld`(2), `pr`(3), `nodest`(4),
`others`(5), `gppr`(6), `gp`(7). Bit-level targeting is supported for
`bitflip` (0-31) and `twobit`.

## 6. Results

| experiment | injections | MASKED | SDC | CRASH | SDC rate |
|---|---|---|---|---|---|
| 001 baseline (no fault) | 10 runs | – | – | – | determ. |
| **002 fault_position** | **100** | **6** | **82** | **12** | **0.82** |
| 003 fault_type | 100 | 6 | 92 | 2 | 0.92 |
| 004 fault_bit | 96 | 3 | 90 | 3 | 0.94 |
| 005 gemm_size | 40 | 0 | 32 | 8 | 0.80 |
| 006 data_pattern | 100 | 4 | 92 | 4 | 0.92 |

The requested `./sdc-lab run --workload gemm --fault-type bitflip
--num-injections 100` corresponds to experiment 002:

```
Total: 100   Masked: 6   SDC: 82   Crash: 12
```

Breakdowns:

- 002 by group: `fp32` s33/c1, `ld` m5/s28, `gp` m1/s21/c11
  (loads mostly corrupt data; GPR/address faults crash).
- 003 by type: `twobit` 1.00 SDC (25/25), `bitflip` 0.96, `random` 0.88,
  `zero` 0.84.
- 004 by FP32 region: exponent 24/24 SDC, sign 3/3 SDC, mantissa m3/s63/c3
  (low mantissa bits 0,2 masked most often).
- 005 by size: 256^3 0.90, 512^3 1.00, 1024^3 0.80, 2048^3 0.50 SDC
  (crash fraction grows with size/instruction exposure).
- 006 by pattern: large 1.00, near_zero 0.95, normal/mixed 0.90, small 0.85.

Generated files per experiment: `summary.csv`, `<dim>_summary.csv`,
`observations.jsonl`, `experiment.json`, `config.yaml`, `report.md`, `raw/`.

## 7. observations.jsonl

One JSON object per injection, with fault metadata and observation metadata
kept separate. Full file (100 lines):
`results/002_fault_position/observations.jsonl` (also 003/004/005/006).

Schema:

```json
{
  "experiment_id": "002_fault_position",
  "seq": 1,
  "workload": "gemm", "M": 1024, "N": 1024, "K": 1024, "dtype": "fp32",
  "fault": {
    "type": "bitflip", "model_id": 0, "group": "fp32", "group_id": 1,
    "kernel": "gemm_fp32(...)", "kernel_count": 0,
    "inst_id": 54636642, "inst_count_profile": 1150287872,
    "inst_fraction": 0.0475, "op_id_seed": 0.529, "bit_id_seed": 0.3599,
    "bit_index": null, "injected": true, "static_site": "FFMA@0x810",
    "register": 24, "mask": "0x800", "before_val": "0x40298243",
    "after_val": "0x40298a43", "opcode": "FFMA", "pc_offset": "0x810",
    "thread": 205571, "block": null, "kernel_error": false
  },
  "observation": {
    "max_abs_error": 0.000488, "mean_abs_error": 4.66e-10,
    "relative_l2_error": 4.47e-08, "corrupted_elements": 1,
    "corrupted_element_ratio": 9.54e-07, "nan_count": 0, "inf_count": 0,
    "total_elements": 1048576, "bitwise_equal": false
  },
  "runtime": { "exit_code": 0, "runtime_sec": 2.74, "timeout": false,
               "crash": false, "crash_reason": null, "stdout_sha256": "..." },
  "gpu": { "temperature_c": "49", "power_w": "90.44",
           "utilization_pct": "0", "memory_used_mb": "25445" },
  "classification": "SDC",
  "reason": "output corrupted (silent data corruption)",
  "labels": { "position": "P0", "group": "fp32", "fault_type": "bitflip" }
}
```

## 8. Extension interfaces

- **Workload** (`core/workload/base.py`): `prepare/run/collect_output/cleanup`
  + registry. GEMM is the only Phase-1 workload; Attention/MLP/Transformer/LLM
  plug in without touching the runner.
- **Injector** (`core/injection/base.py`): `profile/inject` + registry.
  `NVBitFIInjector` is one backend; `SoftwareInjector`/`CustomInjector` can be
  added and selected with `--injector`.
- **Observer** (`core/observer/`): OutputObserver, RuntimeObserver,
  GPUObserver (nvidia-smi, best-effort). New observers (activation, gradient,
  loss, weight, DCGM) are additive.
- **Oracle** (`core/oracle/`): comparator (configurable tolerances) +
  classifier. Thresholds come from the baseline distribution.
- **Experiment kinds** (`core/experiments.py`): `baseline`, `uniform`,
  `fault_position`, `fault_type`, `fault_bit`, `gemm_size`, `data_pattern`;
  add a builder to support a new dimension.
- **InputGenerator** (reserved): `pattern` parameter in `gemm.cu`
  (`normal/small/large/near_zero/mixed`) is the hook for input-aware SDC
  testing.

## Acceptance checklist

```
[PASS] A100 detected
[PASS] GEMM executes
[PASS] Golden reproducible (bit-exact, 10/10)
[PASS] NVBitFI available (built for sm_80)
[PASS] Fault injection works
[PASS] Fault metadata collected
[PASS] Output comparison works
[PASS] MASKED classification works
[PASS] SDC classification works
[PASS] CRASH classification works
[PASS] Results saved as JSONL/CSV
```
