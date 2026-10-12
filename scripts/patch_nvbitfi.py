#!/usr/bin/env python3
"""Apply the gpu-sdc-lab NVBitFI patches (idempotent).

Patches applied:
  1. dedicated `atom` group (ATOM/ATOMS/SUATOM)
  2. dedicated `mma` group (HMMA/IMMA) + fix the HMMA destination-register
     count for sm_80 (a HMMA.16816 accumulates into 4 FP32 registers, but
     upstream extractSize() returns 64 -> only 2). This is required to inject
     into Tensor-Core FP32 accumulators.

  common/arch.h        - add G_ATOM / G_MMA to the group enum (before G_GPPR)
  common/globals.h     - group names + atomInst[]/mmaInst[] + getOpGroupNum()
                         + extractSize() HMMA fix
  injector/inject_funcs.cu - group check + switch case
  scripts/params.py    - IGID_STR names

Usage: patch_nvbitfi.py <path-to-tools/nvbitfi>
"""

import os
import sys


def apply(path, old, new, marker):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if marker in text:
        print("  [skip] %-40s (already patched)" % os.path.basename(path))
        return
    if old not in text:
        raise SystemExit("ERROR: anchor not found in %s:\n%r" % (path, old))
    text = text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  [patch] %s" % os.path.relpath(path))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    fi = sys.argv[1]
    arch = os.path.join(fi, "common", "arch.h")
    glob = os.path.join(fi, "common", "globals.h")
    inj = os.path.join(fi, "injector", "inject_funcs.cu")
    params = os.path.join(fi, "scripts", "params.py")
    for p in (arch, glob, inj, params):
        if not os.path.exists(p):
            raise SystemExit("ERROR: missing %s" % p)

    print("applying atom-group patch to %s" % fi)

    # 1. arch.h: add G_ATOM between G_OTHERS and G_GPPR so that G_GPPR/G_GP
    # stay the last two groups (NVBitFI indexes aggregate counters as
    # NUM_COUNTERS-2 / NUM_COUNTERS-1).
    apply(
        arch,
        "\tG_GPPR, // instructions that write to general purpose and predicate registers\n",
        "\tG_ATOM, // atomic instructions (ATOM/ATOMS/SUATOM)\n"
        "\tG_GPPR, // instructions that write to general purpose and predicate registers\n",
        "G_ATOM",
    )

    # 2. globals.h: group name (same position: after others, before gppr)
    apply(
        glob,
        '"others", "gppr"',
        '"others", "atom", "gppr"',
        '"atom"',
    )

    # 3. globals.h: opcode table after ldInst
    apply(
        glob,
        "int ldInst[] = {\n LD, LDC, LDG, LDL, LDS, SULD, SUST, TLD, TLD4, TLD4S, TLDS\n };\n",
        "int ldInst[] = {\n LD, LDC, LDG, LDL, LDS, SULD, SUST, TLD, TLD4, TLD4S, TLDS\n };\n\n"
        "int atomInst[] = {\n ATOM, ATOMS, SUATOM\n };\n",
        "int atomInst[]",
    )

    # 4. globals.h: classify atom before the generic "others" bucket
    apply(
        glob,
        "\tif (checkOpType(opcode, otherInst, sizeof(otherInst)/sizeof(int))) \n\t\treturn G_OTHERS;\n",
        "\tif (checkOpType(opcode, atomInst, sizeof(atomInst)/sizeof(int)))\n\t\treturn G_ATOM;\n"
        "\tif (checkOpType(opcode, otherInst, sizeof(otherInst)/sizeof(int))) \n\t\treturn G_OTHERS;\n",
        "return G_ATOM",
    )

    # 5. inject_funcs.cu: reject non-atom instructions for igid == G_ATOM
    apply(
        inj,
        "(igid < G_NODEST &&  grp_index != igid))",
        "(igid == G_ATOM && grp_index != G_ATOM) ||\n \t\t\t(igid < G_NODEST &&  grp_index != igid))",
        "G_ATOM && grp_index != G_ATOM",
    )

    # 6. inject_funcs.cu: switch case (atom uses the same group counter)
    apply(
        inj,
        "case G_FP32: // inject into one of the dest reg ",
        "case G_ATOM: // inject into the destination of an atomic instruction\n"
        "\t\tcase G_FP32: // inject into one of the dest reg ",
        "case G_ATOM:",
    )

    # 8. globals.h: normalize modern SASS atomic/reduction opcodes. On sm_70+
    #    global atomics are `ATOMG...` and `REDG...`, which NVBitFI's enum does
    #    not know (they would fall through to index 0 = FADD).
    apply(
        glob,
        "\tstd::string:: size_type pos = opcode.find('.');\n"
        "\tif (pos != std::string::npos) {\n"
        "\t\treturn opcode.substr(0,pos);\n"
        "\t} else {\n"
        "\t\treturn opcode;\n"
        "\t}\n",
        "\tstd::string:: size_type pos = opcode.find('.');\n"
        "\tstd::string t = (pos != std::string::npos) ? opcode.substr(0,pos) : opcode;\n"
        "\tif (t == \"ATOMG\") t = \"ATOM\";\n"
        "\tif (t == \"REDG\") t = \"RED\";\n"
        "\treturn t;\n",
        "ATOMG",
    )

    # 7. params.py: IGID_STR (same order)
    apply(
        params,
        '[ "fp64", "fp32", "ld", "pr", "nodest", "others", "gppr", "gp" ]',
        '[ "fp64", "fp32", "ld", "pr", "nodest", "others", "atom", "gppr", "gp" ]',
        '"atom"',
    )

    # ---------------- mma group (HMMA/IMMA) + HMMA dest fix ----------------
    print("applying mma-group + HMMA-destination patch")

    # 9. arch.h: add G_MMA between G_ATOM and G_GPPR (keep gppr/gp last two)
    apply(
        arch,
        "\tG_ATOM, // atomic instructions (ATOM/ATOMS/SUATOM)\n"
        "\tG_GPPR, // instructions that write to general purpose and predicate registers\n",
        "\tG_ATOM, // atomic instructions (ATOM/ATOMS/SUATOM)\n"
        "\tG_MMA, // tensor-core matrix instructions (HMMA/IMMA)\n"
        "\tG_GPPR, // instructions that write to general purpose and predicate registers\n",
        "G_MMA",
    )

    # 10. globals.h: group name
    apply(
        glob,
        '"others", "atom", "gppr"',
        '"others", "atom", "mma", "gppr"',
        '"mma"',
    )

    # 11. globals.h: opcode table after atomInst
    apply(
        glob,
        "int atomInst[] = {\n ATOM, ATOMS, SUATOM\n };\n",
        "int atomInst[] = {\n ATOM, ATOMS, SUATOM\n };\n\n"
        "int mmaInst[] = {\n IMMA, HMMA\n };\n",
        "int mmaInst[]",
    )

    # 12. globals.h: classify mma before the generic "others" bucket
    apply(
        glob,
        "\tif (checkOpType(opcode, otherInst, sizeof(otherInst)/sizeof(int))) \n"
        "\t\treturn G_OTHERS;\n",
        "\tif (checkOpType(opcode, mmaInst, sizeof(mmaInst)/sizeof(int)))\n"
        "\t\treturn G_MMA;\n"
        "\tif (checkOpType(opcode, otherInst, sizeof(otherInst)/sizeof(int))) \n"
        "\t\treturn G_OTHERS;\n",
        "return G_MMA",
    )

    # 13. globals.h: HMMA writes 4 FP32 accumulator regs on sm_80 (not 2)
    apply(
        glob,
        '\tif (opcode.find("HMMA") != std::string::npos)\n\t\treturn 64;\n',
        '\tif (opcode.find("HMMA") != std::string::npos)\n'
        '\t\treturn 128; // sm_80: 4 FP32 accumulator registers\n',
        "return 128; // sm_80",
    )

    # 14. inject_funcs.cu: reject non-mma instructions for igid == G_MMA
    apply(
        inj,
        "(igid == G_ATOM && grp_index != G_ATOM) ||\n",
        "(igid == G_ATOM && grp_index != G_ATOM) ||\n"
        " \t\t\t(igid == G_MMA && grp_index != G_MMA) ||\n",
        "G_MMA && grp_index != G_MMA",
    )

    # 15. inject_funcs.cu: switch case (mma uses the dest-register path)
    apply(
        inj,
        "\t\tcase G_ATOM: // inject into the destination of an atomic instruction\n",
        "\t\tcase G_MMA: // inject into an HMMA/IMMA accumulator register\n"
        "\t\tcase G_ATOM: // inject into the destination of an atomic instruction\n",
        "case G_MMA:",
    )

    # 16. params.py: IGID_STR names (same order)
    apply(
        params,
        '[ "fp64", "fp32", "ld", "pr", "nodest", "others", "atom", "gppr", "gp" ]',
        '[ "fp64", "fp32", "ld", "pr", "nodest", "others", "atom", "mma", "gppr", "gp" ]',
        '"mma"',
    )

    print("atom + mma patches applied.")


if __name__ == "__main__":
    main()
