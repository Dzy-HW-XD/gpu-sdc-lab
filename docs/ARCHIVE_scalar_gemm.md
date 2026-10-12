# Archive: scalar (CUDA Core) GEMM era

The experiments **015–019** (and the earlier 001–007) targeted the **scalar FP32
CUDA-Core path** of a naive `C=A×B` kernel: NVBitFI faults were injected into
the destination registers of `FFMA`/`FADD`/`FMUL` instructions
(`core/regclass.py` labelled them accumulator / temporary).

These results and figures are **kept for reference only** and are **not**
re-run. The current direction is the Tensor-Core path (BF16 × BF16, FP32
accumulation, cuBLAS, HMMA fault injection) — see `docs/TC_BF16_HMMA.md` and the
experiments `020–023`.

Archived artifacts:
* `results/015_input_bit_gemm/`, `results/016_*` … `results/019_*` — scalar
  `input × bit/instruction/size/position/register` campaign (raw preserved in
  `results/_raw_archive/raw_campaign.tar.gz`, not in git).
* `results/_campaign/`, `results/_tolerance/` — scalar-era summaries and
  tolerance sensitivity.
* `docs/INPUT_FAMILIES.md` — the synthetic input families used then.

The framework that supported them (workload abstraction, runner, oracle,
observer, taxonomy, sharding, tolerance sweep) is unchanged and reused by the
Tensor-Core experiments.
