"""Machine capacity before heavy stages.

Transcription and summarization each need several gigabytes. On a 16 GB Mac
the difference between a finished job and a swap storm is what else is running
at the time, so the pipeline asks the operating system first and refuses to
start work the machine cannot afford.

The checks read the kernel's own signals: the memory pressure level, the
reclaimable page count, swap usage, and the load average.
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

# A window peaks near 1.3 GB on the default MLX engine and near 3.4 GB with
# torch; below this the machine is already
# short of memory before the model is loaded.
TRANSCRIBE_HEADROOM_GB = 2.5
# The 9B summary model is ~6 GB of weights plus its cache.
SUMMARY_HEADROOM_GB = 6.0
# Load average above this per CPU means the machine is already busy.
BUSY_LOAD_PER_CPU = 1.5
# Pressure levels reported by kern.memorystatus_vm_pressure_level.
PRESSURE_NORMAL = 1
PRESSURE_WARNING = 2
PRESSURE_CRITICAL = 4

_PRESSURE_NAMES = {
    PRESSURE_NORMAL: "normal",
    PRESSURE_WARNING: "warning",
    PRESSURE_CRITICAL: "critical",
}


class CapacityError(RuntimeError):
    """The machine cannot afford a stage right now."""


@dataclass(frozen=True)
class Capacity:
    pressure: int
    headroom_gb: float
    swap_used_gb: float
    load_1m: float
    cpus: int

    @property
    def pressure_name(self) -> str:
        return _PRESSURE_NAMES.get(self.pressure, f"level {self.pressure}")

    @property
    def load_per_cpu(self) -> float:
        return self.load_1m / max(self.cpus, 1)

    def describe(self) -> str:
        headroom = "unlimited" if self.headroom_gb == float("inf") else f"{self.headroom_gb:.1f} GB"
        return (
            f"memory pressure {self.pressure_name}, "
            f"{headroom} reclaimable, "
            f"swap {self.swap_used_gb:.1f} GB used, "
            f"load {self.load_1m:.1f} on {self.cpus} CPUs"
        )


def _tool(name: str) -> str:
    """Absolute path for a macOS tool, so no PATH entry can shadow it."""
    return shutil.which(name) or f"/usr/sbin/{name}"


def _sysctl(name: str) -> str:
    result = subprocess.run(  # noqa: S603
        [_tool("sysctl"), "-n", name], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _pressure_level() -> int:
    try:
        return int(_sysctl("kern.memorystatus_vm_pressure_level"))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return PRESSURE_NORMAL


def _page_headroom_gb() -> float:
    """Free plus reclaimable memory, in gigabytes, from vm_stat."""
    try:
        result = subprocess.run(  # noqa: S603
            [_tool("vm_stat")], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return _fallback_headroom_gb()
    text = result.stdout
    page_size_match = re.search(r"page size of (\d+) bytes", text)
    if not page_size_match:
        return _fallback_headroom_gb()
    page_size = int(page_size_match.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages purgeable"):
        match = re.search(rf"^{label}:\s+(\d+)", text, re.MULTILINE)
        if match:
            pages += int(match.group(1))
    return pages * page_size / 1024**3


def _fallback_headroom_gb() -> float:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return float("inf")
    return pages * page_size / 1024**3


def _swap_used_gb() -> float:
    try:
        match = re.search(r"used = ([\d.]+)M", _sysctl("vm.swapusage"))
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    return float(match.group(1)) / 1024 if match else 0.0


def measure() -> Capacity:
    load_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    cpus = os.cpu_count() or 1
    if sys.platform != "darwin":
        # The signals are macOS-specific, and capturing only runs on macOS.
        # Elsewhere the pipeline is never gated on capacity.
        return Capacity(PRESSURE_NORMAL, float("inf"), 0.0, load_1m, cpus)
    return Capacity(
        pressure=_pressure_level(),
        headroom_gb=_page_headroom_gb(),
        swap_used_gb=_swap_used_gb(),
        load_1m=load_1m,
        cpus=cpus,
    )


def shortfall(capacity: Capacity, need_gb: float) -> str | None:
    """Returns why the machine cannot afford `need_gb`, or None when it can."""
    if capacity.pressure >= PRESSURE_WARNING:
        return f"memory pressure is {capacity.pressure_name}"
    if capacity.headroom_gb < need_gb:
        return (
            f"only {capacity.headroom_gb:.1f} GB of memory is reclaimable "
            f"and this step needs about {need_gb:g} GB"
        )
    return None


def require(need_gb: float, stage: str, force: bool = False) -> Capacity:
    """Measures capacity and raises unless the stage can afford to run."""
    capacity = measure()
    reason = shortfall(capacity, need_gb)
    if reason and not force:
        raise CapacityError(
            f"Not starting {stage}: {reason}. {capacity.describe()}. "
            "Close applications, then retry; or pass --force to run anyway."
        )
    return capacity


def warn_on_load(capacity: Capacity) -> None:
    """CPU contention slows every stage; it is reported, never enforced."""
    if capacity.load_per_cpu > BUSY_LOAD_PER_CPU and sys.stderr.isatty():
        print(
            f"Note: the machine is busy ({capacity.describe()}); stages will be slower.",
            file=sys.stderr,
        )
