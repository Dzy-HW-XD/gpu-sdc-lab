# NVBitFI fault-injection model

Backend: [NVlabs/nvbitfi](https://github.com/NVlabs/nvbitfi) on top of
[NVlabs/NVBit](https://github.com/NVlabs/NVBit) (v1.8.1).

NVBitFI instruments a GPU application via `LD_PRELOAD` and injects one error
into the destination register of one chosen dynamic thread-instruction. It
supports the fault models below (ids match `scripts/params.py` in NVBitFI):

| model      | id | meaning                              |
|------------|----|--------------------------------------|
| `bitflip`  | 0  | single bit flip in a register        |
| `twobit`   | 1  | two adjacent bits flipped            |
| `random`   | 2  | random value written to a register   |
| `zero`     | 3  | register zeroed                      |

Instruction groups (`groupID`):

| name    | id | name    | id |
|---------|----|---------|----|
| `fp64`  | 0  | `atom`* | 6  |
| `fp32`  | 1  | `gppr`  | 7  |
| `ld`    | 2  | `gp`    | 8  |
| `pr`    | 3  |         |    |
| `nodest`| 4  |         |    |
| `others`| 5  |         |    |

\* `atom` is a **lab-added** group (see below), inserted between `others` and
`gppr` so that NVBitFI's aggregate counters (`NUM_COUNTERS-2`/`-1` = gppr/gp)
stay valid.

## Injection site selection

A fault site is `(kernel_name, kernel_invocation, group, dynamic_inst_id)`.
The dynamic instruction id is the ordinal of the instruction **within the
selected group**, counted per thread over the whole kernel invocation. The
per-group totals come from a one-time `profiler.so` run of the exact workload.

The bit (for `bitflip`) is chosen through `bitIDSeed`:
`bit = int(32 * bitIDSeed)` (single) or `int(31 * bitIDSeed)` (two-bit).
Setting `bitIDSeed = (bit + 0.5) / 32` therefore targets bit `bit` exactly.
FP32 bit layout: bit 31 sign, bits 23-30 exponent, bits 0-22 mantissa.

## Lab extension: dedicated `atom` group

To target atomic instructions (ATOM/ATOMS) exactly instead of sampling the
shared `others` bucket, `scripts/patch_nvbitfi.py` adds one dedicated group.
The patch is idempotent and applied by `scripts/setup_nvbitfi.sh`. It touches:

```
common/arch.h            G_ATOM between G_OTHERS and G_GPPR
common/globals.h         instGrouptNames + atomInst[] + getOpGroupNum()
                         + extractInstType(): normalize ATOMG -> ATOM, REDG -> RED
injector/inject_funcs.cu group check + switch case (uses currCounter1)
scripts/params.py        IGID_STR name
```

The `extractInstType` normalization is required because on `sm_70+` SASS emits
`ATOMG...`/`REDG...` (e.g. `ATOMG.E.ADD.F32`) while NVBitFI's enum only knows
`ATOM`/`ATOMS`; without it those opcodes fall through to index 0 (`FADD`).

Adding further groups (`mma`, `fp16`, `int`, ...) follows the same recipe.
`core/taxonomy.py` is the lab-side declaration of which leaf maps to which
group / opcode set.

## A100 / CUDA-13 compatibility patches

NVBitFI is a 2020 tool tested on NVBit 1.5.5 / CUDA 11.2. On A100 with driver
580 (CUDA 13.0) two source patches are required (applied by
`scripts/setup_nvbitfi.sh`):

1. **profiler** `inject_funcs.cu`: `ballot()` is removed for `sm_70+`.
   Replace `ballot(1)` / `ballot(pred)` with `__activemask()` /
   `__ballot_sync(__activemask(), pred)`.

2. **injector** `inject_funcs.cu`: device-side `printf()` and `assert()`
   require CUDA >= 13.1. On older CUDA they produce
   `CUDA_ERROR_INVALID_SOURCE` while the instrumented kernel loads. They are
   compiled out (they are debug-only).

Tools must be built with `ARCH=80` (A100). No changes to the system
driver/CUDA installation are required.

## Fault metadata

After each injection the injector writes `nvbitfi-injection-log-temp.txt`
containing the injected register (`regNo`), opcode, pcOffset, thread id,
bit mask and before/after values. The lab parses this into the `fault` block of
each result record.
