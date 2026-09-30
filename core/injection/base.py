"""Fault-injection abstraction.

The framework depends only on this interface. NVBitFI is the Phase-1 backend;
SoftwareInjector / CustomInjector / hardware injectors can be added later
without touching workloads, oracles or observers.
"""

# Group ids must match the (patched) NVBitFI common/arch.h enum. The `atom`
# group is inserted between `others` and `gppr` so NVBitFI's aggregate counters
# (NUM_COUNTERS-2 = gppr, NUM_COUNTERS-1 = gp) stay valid.
GROUP_IDS = {
    "fp64": 0,
    "fp32": 1,
    "ld": 2,
    "pr": 3,
    "nodest": 4,
    "others": 5,
    "atom": 6,   # added by scripts/patch_nvbitfi.py (ATOM/ATOMS/SUATOM)
    "gppr": 7,
    "gp": 8,
}
GROUP_NAMES = {v: k for k, v in GROUP_IDS.items()}

# NVBitFI bit-flip model ids (scripts/params.py)
BITFLIP_MODELS = {
    "bitflip": 0,   # FLIP_SINGLE_BIT
    "twobit": 1,    # FLIP_TWO_BITS
    "random": 2,    # RANDOM_VALUE
    "zero": 3,      # ZERO_VALUE
}
MODEL_NAMES = {v: k for k, v in BITFLIP_MODELS.items()}


class FaultSpec(object):
    """A single instruction-level fault to inject."""

    def __init__(self, group, kernel_name, kernel_count, inst_id,
                 op_id_seed, bit_flip_model, bit_id_seed,
                 fault_type=None, bit_index=None):
        self.group = int(group)
        self.group_name = GROUP_NAMES.get(self.group, str(group))
        self.kernel_name = kernel_name
        self.kernel_count = int(kernel_count)
        self.inst_id = int(inst_id)
        self.op_id_seed = float(op_id_seed)
        self.bit_flip_model = int(bit_flip_model)
        self.bit_id_seed = float(bit_id_seed)
        self.fault_type = fault_type or MODEL_NAMES.get(self.bit_flip_model, "?")
        self.bit_index = bit_index

    def injection_info_text(self):
        return "%d\n%d\n%s\n%d\n%d\n%.10f\n%.10f\n" % (
            self.group, self.bit_flip_model, self.kernel_name,
            self.kernel_count, self.inst_id, self.op_id_seed, self.bit_id_seed,
        )

    def to_dict(self):
        return {
            "type": self.fault_type,
            "model_id": self.bit_flip_model,
            "group": self.group_name,
            "group_id": self.group,
            "kernel": self.kernel_name,
            "kernel_count": self.kernel_count,
            "inst_id": self.inst_id,
            "op_id_seed": self.op_id_seed,
            "bit_id_seed": self.bit_id_seed,
            "bit_index": self.bit_index,
        }


class FaultInjector(object):
    name = "base"

    def __init__(self, lab):
        self.lab = lab

    def profile(self, workload):
        raise NotImplementedError

    def inject(self, workload, spec, run_dir, device=0, timeout=None):
        raise NotImplementedError

    def available(self):
        return False

    def describe(self):
        return {"name": self.name, "available": self.available()}
