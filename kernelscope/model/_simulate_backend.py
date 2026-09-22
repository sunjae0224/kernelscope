"""Optional, locally compiled CPU simulator; import never invokes a compiler."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np

_LOCK = threading.Lock()
_FUNCTION = None
_LIBRARY = None  # Keep the shared library alive while ctypes calls it.
_METADATA = None
_FLAGS = ("-O3", "-std=c11", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off")


def _cache_directory():
    requested = os.environ.get("KERNELSCOPE_SIMULATOR_CACHE")
    project = Path(__file__).resolve().parents[2]
    candidate = Path(requested).expanduser() if requested else project / ".cache/kernelscope/simulator"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        # A real create catches readonly filesystems, including installed wheels.
        with tempfile.TemporaryFile(dir=candidate):
            pass
        return candidate
    except OSError:
        if requested:
            raise
        return Path(tempfile.mkdtemp(prefix="kernelscope-simulator-"))


def prepare_native():
    """Return (callable or None, setup metadata), caching success and failure."""
    global _FUNCTION, _LIBRARY, _METADATA
    if _METADATA is not None:
        return _FUNCTION, dict(_METADATA)
    with _LOCK:
        if _METADATA is not None:
            return _FUNCTION, dict(_METADATA)
        started = time.perf_counter()
        source = Path(__file__).with_name("_simulate_native.c")
        metadata = {"backend": "python_reference", "native_available": False,
                    "built_this_process": False, "cache_hit": False, "setup_us": 0.0,
                    "compiler": None, "compiler_version_sha256": None,
                    "source_sha256": None, "cache_path": None, "error": None,
                    "compile_flags": list(_FLAGS)}
        temporary = None
        try:
            source_bytes = source.read_bytes()
            metadata["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
            compiler = shlex.split(os.environ.get("CC", "cc"))
            executable = shutil.which(compiler[0]) if compiler else None
            if not executable:
                raise RuntimeError("no C compiler found; using the NumPy reference simulator")
            compiler[0] = executable
            probe = subprocess.run(compiler + ["--version"], capture_output=True, text=True,
                                   check=True, timeout=10)
            metadata["compiler"] = compiler
            metadata["compiler_version_sha256"] = hashlib.sha256(probe.stdout.encode()).hexdigest()
            identity = source_bytes + repr((compiler, probe.stdout, _FLAGS, platform.machine(), platform.system())).encode()
            destination = _cache_directory() / ("simulate_" + hashlib.sha256(identity).hexdigest()[:24] + ".so")
            metadata["cache_path"] = str(destination)
            metadata["cache_hit"] = destination.is_file()
            if not destination.is_file():
                temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
                command = compiler + list(_FLAGS) + [str(source), "-o", str(temporary), "-lm"]
                build = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if build.returncode:
                    raise RuntimeError("C simulator compilation failed: " + build.stderr[-2000:])
                os.replace(temporary, destination)
                metadata["built_this_process"] = True
            library = ctypes.CDLL(str(destination))
            function = library.kernelscope_simulate
            function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
                                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                 np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
                                 *([ctypes.c_double] * 6), ctypes.c_size_t,
                                 np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
            function.restype = ctypes.c_int
            _LIBRARY, _FUNCTION = library, function
            metadata.update(backend="native_c", native_available=True)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            metadata["error"] = str(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        metadata["setup_us"] = (time.perf_counter() - started) * 1e6
        _METADATA = metadata
        return _FUNCTION, dict(metadata)
