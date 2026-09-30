"""Shared helpers: subprocess execution, hashing, JSON I/O, paths."""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import time


def detect_cuda_home(explicit=None):
    """Locate a CUDA toolkit without hardcoding a path.

    Order: explicit value -> $CUDA_HOME/$CUDA_PATH -> nvcc on PATH ->
    /usr/local/cuda -> newest /usr/local/cuda-*.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(var)
        if v:
            candidates.append(v)
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    candidates.append("/usr/local/cuda")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "bin", "nvcc")):
            return os.path.realpath(c)
    return candidates[0] if candidates else "/usr/local/cuda"


def run_cmd(cmd, cwd=None, env=None, timeout=None, extra_env=None):
    """Run a command, capturing output. Returns a dict.

    cmd: list of argv (preferred) or a shell string.
    extra_env: merged on top of env or os.environ.
    """
    use_env = dict(os.environ)
    if env:
        use_env.update({k: v for k, v in env.items() if v is not None})
    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                use_env.pop(k, None)
            else:
                use_env[k] = v

    shell = isinstance(cmd, str)
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=use_env,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                out, err = proc.communicate(timeout=10)
            except Exception:
                out, err = b"", b""
            code = -9
    except FileNotFoundError as e:
        return {
            "cmd": cmd,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(e),
            "runtime_sec": 0.0,
            "timeout": False,
        }
    runtime = time.time() - t0
    return {
        "cmd": cmd if not shell else [cmd],
        "exit_code": code,
        "stdout": out.decode("utf-8", "replace"),
        "stderr": err.decode("utf-8", "replace"),
        "runtime_sec": runtime,
        "timeout": timed_out,
    }


def sha256_file(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text, n=12):
    return sha256_text(text)[:n]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, obj, indent=2):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False)
        f.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, obj):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
