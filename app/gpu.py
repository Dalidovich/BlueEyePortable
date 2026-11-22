from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("blueeye.app")

BF16_MIN_CUDA_MAJOR = 8


@dataclass
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int
    capability: tuple[int, int]
    compatible: bool
    bf16_supported: bool

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"

    @property
    def capability_label(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"

    @property
    def vram_gib(self) -> float:
        return self.total_memory_bytes / 2**30


@dataclass
class DeviceReport:
    torch_version: str
    cuda_runtime: str | None
    cuda_available: bool
    built_architectures: list[str] = field(default_factory=list)
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda_error: str | None = None

    @property
    def compatible_gpus(self) -> list[GpuInfo]:
        return [gpu for gpu in self.gpus if gpu.compatible]

    def gpu_for(self, device: str) -> GpuInfo | None:
        return next((gpu for gpu in self.gpus if gpu.device == device), None)


def _parse_arch(arch: str) -> tuple[int, int] | None:
    if not arch.startswith("sm_"):
        return None
    digits = arch[3:].rstrip("af")
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def is_capability_compatible(capability: tuple[int, int], built_architectures: list[str]) -> bool:
    major, minor = capability
    for arch in built_architectures:
        parsed = _parse_arch(arch)
        if parsed and parsed[0] == major and parsed[1] <= minor:
            return True
    return False


def _probe_gpu(index: int) -> None:
    with torch.cuda.device(index):
        x = torch.ones((64, 64), device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        if float((x @ x).sum().item()) != 64.0 * 64.0 * 64.0:
            raise RuntimeError("CUDA self-test returned a wrong result")


def detect_devices() -> DeviceReport:
    report = DeviceReport(torch_version=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=False)
    if torch.version.cuda is None:
        report.cuda_error = "This PyTorch build has no CUDA support."
        return report
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            available = torch.cuda.is_available()
            report.built_architectures = list(torch.cuda.get_arch_list()) if available else []
            count = torch.cuda.device_count() if available else 0
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                capability = (props.major, props.minor)
                compatible = is_capability_compatible(capability, report.built_architectures)
                report.gpus.append(GpuInfo(
                    index=index,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    capability=capability,
                    compatible=compatible,
                    bf16_supported=compatible and capability[0] >= BF16_MIN_CUDA_MAJOR,
                ))
    except Exception as e:
        report.cuda_error = f"CUDA initialisation failed: {type(e).__name__}: {e}"
        logger.exception("CUDA initialisation failed")
        return report

    if not available:
        report.cuda_error = "No CUDA device available (no NVIDIA GPU, or the NVIDIA driver is missing or too old)."
        return report

    for gpu in report.gpus:
        if not gpu.compatible:
            logger.warning("GPU %s (%s) is not supported by this PyTorch build (%s)",
                           gpu.name, gpu.capability_label, " ".join(report.built_architectures))
            continue
        try:
            _probe_gpu(gpu.index)
        except Exception as e:
            gpu.compatible = False
            gpu.bf16_supported = False
            logger.warning("GPU %s failed the CUDA self-test: %s", gpu.name, e)
            report.cuda_error = f"{gpu.name}: CUDA self-test failed: {type(e).__name__}: {e}"

    report.cuda_available = bool(report.compatible_gpus)
    if not report.cuda_available and report.cuda_error is None:
        report.cuda_error = "No GPU compatible with this PyTorch build was found."
    return report


def cpu_bf16_supported() -> bool:
    try:
        return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:
        return False


def choose_device(report: DeviceReport, preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    if preference.startswith("cuda"):
        gpu = report.gpu_for(preference)
        if gpu is not None and gpu.compatible:
            return preference
        logger.warning("Preferred device %s is unavailable; selecting automatically", preference)
    candidates = report.compatible_gpus
    if not candidates:
        return "cpu"
    return max(candidates, key=lambda gpu: gpu.total_memory_bytes).device


def bf16_available(report: DeviceReport, device: str) -> bool:
    if device == "cpu":
        return cpu_bf16_supported()
    gpu = report.gpu_for(device)
    return gpu is not None and gpu.bf16_supported


def device_label(report: DeviceReport, device: str) -> str:
    gpu = report.gpu_for(device)
    if gpu is None:
        return "CPU"
    return f"{gpu.name} ({device})"
