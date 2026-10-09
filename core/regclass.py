"""Static SASS register-class labelling (accumulator vs temporary).

For the register-type dimension of the SDC study we need to know, for the
*faulted* dynamic instruction, whether the destination register plays the role
of a long-lived accumulator or of a short-lived temporary.

The NVBitFI fault metadata records the injected instruction's `pc_offset` and
`opcode`, so we can look the instruction up in a static disassembly of the
workload binary and classify it syntactically:

  * FFMA Rd, Ra, Rb, Rc  -> accumulator if Rd == Rc (running sum), else temp
  * FADD/FMUL Rd, Ra, Rb -> accumulator if Rd is also an operand (in-place
                            update), otherwise temporary

This is a best-effort *static* heuristic (no liveness analysis). It is exact
for the simple, fully unrolled GEMM inner loop but should be treated as an
approximation for more complex kernels.
"""

import os
import re

from core import util

FP32_OPS = ("FFMA", "FADD", "FMUL", "FFMA32I", "FADD32I", "FMUL32I")

_REG = re.compile(r"\b(R\d+|UR\d+|SR\d+|P\d+|RZ|URZ|SRZ)\b")
_LINE = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+([A-Z][A-Z0-9._]*)\s+([^;]*);")


def _norm_pc(s):
    """Normalize an NVBitFI pcOffset ('0x4c0', '4c0', '1216') to an int."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s)
    except ValueError:
        try:
            return int(s, 16)
        except ValueError:
            return None


def _first_reg(text):
    m = _REG.search(text or "")
    return m.group(1) if m else None


def classify_instruction(opcode, operands_text):
    """Return 'accumulator' | 'temporary' | 'other' for one SASS instruction."""
    base = opcode.split(".")[0]
    if base not in FP32_OPS:
        return "other"
    parts = [p.strip() for p in operands_text.split(",") if p.strip()]
    if not parts:
        return "other"
    dest = _first_reg(parts[0])
    if dest is None or not dest.startswith("R"):
        return "other"
    srcs = [_first_reg(p) for p in parts[1:]]
    srcs = [s for s in srcs if s]
    if dest in srcs:
        return "accumulator"
    return "temporary"


def parse_sass(text):
    """Parse cuobjdump -sass output -> {pc_int: {opcode, regclass}}."""
    table = {}
    for line in text.splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        pc = int(m.group(1), 16)
        opcode = m.group(2)
        operands = m.group(3)
        if opcode.split(".")[0] not in FP32_OPS:
            continue
        table[pc] = {
            "opcode": opcode,
            "regclass": classify_instruction(opcode, operands),
            "sass": line.strip(),
        }
    return table


def disassemble(binary, cuda_home):
    cuobjdump = os.path.join(cuda_home, "bin", "cuobjdump")
    if not os.path.exists(cuobjdump):
        cuobjdump = "cuobjdump"
    res = util.run_cmd([cuobjdump, "-sass", binary], timeout=120)
    return res.get("stdout", "")


def load_table(binary, cuda_home, cache_dir=None):
    """Disassemble + parse, optionally caching the table as JSON."""
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "_sass")
    util.ensure_dir(cache_dir)
    key = "%s_%d" % (os.path.basename(binary), int(os.path.getmtime(binary)))
    cache = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cache):
        return {int(k): v for k, v in util.read_json(cache).items()}
    table = parse_sass(disassemble(binary, cuda_home))
    util.write_json(cache, {str(k): v for k, v in table.items()})
    return table


def regclass_for(pc_offset, table):
    pc = _norm_pc(pc_offset)
    if pc is None or not table:
        return None
    rec = table.get(pc)
    return rec.get("regclass") if rec else None
