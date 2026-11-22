from __future__ import annotations

BYTES_PER_MIB = 2**20


def format_float(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def format_mib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / BYTES_PER_MIB:.0f}"
