"""Instruction taxonomy: a small tree that is the single source of truth for
"what can be selected as a fault target".

Each leaf maps to a NVBitFI instruction group and (optionally) an opcode
whitelist:

  * leaves with a dedicated NVBitFI group (fp32/fp64/ld/pr/atom/others/gppr/gp)
    are selected exactly: inst_id is sampled within that group's dynamic stream.
  * leaves whose group is a shared bucket ("others") carry an opcode whitelist;
    they are *opcode-filtered*: the site is sampled inside the bucket and the
    result is only counted when the hit opcode is in the whitelist.

Selection is a tree walk; a node can be a top-level category, a leaf name, or a
path like "memory/atomic". A subtree expands to all its leaves.
"""

from core.injection.base import GROUP_IDS

# opcode whitelists (names match NVBitFI common/arch.h enum)
_FP16 = ["HADD2", "HADD2_32I", "HFMA2", "HFMA2_32I", "HMUL2", "HMUL2_32I",
         "HSET2", "HSETP2"]
_MMA = ["HMMA", "IMMA"]
_ATOMIC = ["ATOM", "ATOMS", "SUATOM"]
_INT = ["IADD", "IADD3", "IADD32I", "IMAD", "IMAD32I", "IMADSP", "IMNMX",
        "IMUL", "IMUL32I", "ISCADD", "ISCADD32I", "ISET", "LEA", "LOP", "LOP3",
        "LOP32I", "PLOP3", "POPC", "SHF", "SHL", "SHR", "XMAD", "BFE", "BFI",
        "BMSK", "BREV", "FLO", "ICMP", "IDP", "IDP4A"]

TAXONOMY = {
    "arithmetic": {
        "fp64": {"group": "fp64"},
        "fp32": {"group": "fp32"},
        "fp16": {"group": "others", "opcodes": _FP16},
        "int": {"group": "others", "opcodes": _INT},
        "mma": {"group": "others", "opcodes": _MMA},
    },
    "memory": {
        "load": {"group": "ld"},
        "atomic": {"group": "atom"},
        "store": {"group": None},          # no destination register -> not injectable
    },
    "control": {
        "predicate": {"group": "pr"},
        "nodest": {"group": None},         # branch / RED / ST / ... -> not injectable
    },
    "meta": {
        "gppr": {"group": "gppr"},
        "gp": {"group": "gp"},
        "other": {"group": "others"},
    },
}

# groups NVBitFI provides as dedicated buckets
DEDICATED_GROUPS = {"fp64", "fp32", "ld", "pr", "others", "gppr", "gp", "atom"}

# convenience aliases (NVBitFI group names that differ from the leaf name)
ALIASES = {
    "ld": "memory/load",
    "others": "meta/other",
    "pr": "control/predicate",
}


def _flatten():
    leaves = {}
    for category, subs in TAXONOMY.items():
        for name, spec in subs.items():
            group = spec.get("group")
            opcodes = spec.get("opcodes")
            leaves["%s/%s" % (category, name)] = {
                "path": "%s/%s" % (category, name),
                "category": category,
                "name": name,
                "group": group,
                "opcodes": opcodes,
                "targeting": "exact" if opcodes is None else "opcode-filtered",
                "injectable": group is not None and group in GROUP_IDS,
            }
    return leaves


LEAVES = _flatten()


def all_leaves():
    return list(LEAVES.values())


def resolve(node):
    """Resolve a node name/path/subtree into a list of leaf dicts."""
    if node is None:
        return []
    if isinstance(node, (list, tuple)):
        out = []
        for n in node:
            out.extend(resolve(n))
        return out
    key = str(node).strip().replace(":", "/").replace(".", "/").strip("/")
    if key == "" or key == "*":
        return all_leaves()
    if key in ALIASES:
        key = ALIASES[key]
    if key in TAXONOMY:  # top-level category
        return [LEAVES[p] for p in LEAVES if LEAVES[p]["category"] == key]
    if key in LEAVES:  # exact path
        return [LEAVES[key]]
    # bare leaf name
    hits = [LEAVES[p] for p in LEAVES if LEAVES[p]["name"] == key]
    if hits:
        return hits
    # unique suffix match on path
    hits = [LEAVES[p] for p in LEAVES if p.endswith("/" + key)]
    if hits:
        return hits
    raise KeyError("unknown taxonomy node '%s' (try one of: %s)"
                   % (node, ", ".join(sorted(LEAVES))))


def printable():
    lines = []
    for category in TAXONOMY:
        lines.append(category)
        for p in sorted(LEAVES):
            leaf = LEAVES[p]
            if leaf["category"] != category:
                continue
            tag = leaf["group"] or "not-injectable"
            extra = "" if leaf["opcodes"] is None else " [%d opcodes]" % len(leaf["opcodes"])
            lines.append("  %-10s -> group=%-8s (%s)%s"
                         % (leaf["name"], tag, leaf["targeting"], extra))
    return "\n".join(lines)
