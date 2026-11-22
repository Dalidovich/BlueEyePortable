from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from settings import PRECISIONS, RESULTS_CSV_PATH, Settings, ensure_directories, isolate_huggingface_environment

PROGRESS_INTERVAL_SECONDS = 1.0
LABEL_CHOICES = ("safe", "suggestive", "explicit")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def format_optional(value: float | None, pattern: str) -> str:
    return "-" if value is None else pattern.format(value)


def load_engine(device_preference: str, precision: str):
    from classifier import BlueEyeEngine, ModelNotReadyError
    from gpu import bf16_available, choose_device, detect_devices, device_label
    from model_manager import check_model

    report = detect_devices()
    device = choose_device(report, device_preference)
    if precision == "bf16" and not bf16_available(report, device):
        print(f"ERROR: bf16 is not available on {device_label(report, device)}. Use --precision fp32.",
              file=sys.stderr)
        return None
    try:
        engine = BlueEyeEngine.load(device, precision)
    except ModelNotReadyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None
    gpu = report.gpu_for(device)
    revision = check_model(full_hash=False).revision
    print(f"Device: {device_label(report, device)}  precision: {precision}")
    return engine, (gpu.name if gpu else None), revision


def command_devices(args: argparse.Namespace) -> int:
    from gpu import bf16_available, choose_device, detect_devices, device_label
    from model_manager import check_model

    report = detect_devices()
    print(f"PyTorch: {report.torch_version}")
    print(f"CUDA runtime: {report.cuda_runtime or 'none'}")
    print(f"CUDA available: {report.cuda_available}")
    if report.cuda_error:
        print(f"CUDA note: {report.cuda_error}")
    for gpu in report.gpus:
        print(f"GPU {gpu.index}: {gpu.name}  VRAM {gpu.vram_gib:.1f} GiB  {gpu.capability_label}  "
              f"compatible={gpu.compatible}  bf16={bf16_available(report, gpu.device)}")
    print(f"CPU bf16: {bf16_available(report, 'cpu')}")
    print(f"Chosen device: {device_label(report, choose_device(report, 'auto'))}")
    status = check_model(full_hash=False)
    print(f"Model: {status.message}" + (f" (revision {status.revision})" if status.revision else ""))
    return 0 if status.present else 1


def command_scan(args: argparse.Namespace) -> int:
    from database import Database
    from scanner import InsufficientDiskSpaceError, Scanner

    loaded = load_engine(args.device, args.precision)
    if loaded is None:
        return 2
    engine, gpu_name, revision = loaded
    stop_flag = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        if not stop_flag["requested"]:
            print("\nStop requested; finishing the current batch...")
        stop_flag["requested"] = True

    last_print = {"time": 0.0, "phase": ""}

    def print_progress(p) -> None:
        now = time.monotonic()
        if p.phase == last_print["phase"] and p.phase != "done" and now - last_print["time"] < PROGRESS_INTERVAL_SECONDS:
            return
        last_print["time"] = now
        last_print["phase"] = p.phase
        rate = format_optional(p.images_per_second, "{:.1f} img/s")
        print(f"[{p.phase}] {p.done}/{p.total}  processed={p.processed} reused={p.reused} "
              f"skipped={p.skipped} errors={p.errors}  {rate}")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    db = Database()
    try:
        scanner = Scanner(db, engine, gpu_name, revision)
        summary = scanner.run(Path(args.folder), not args.no_recursive, args.batch_size, force=args.force,
                              progress=print_progress, should_stop=lambda: stop_flag["requested"])
    except (FileNotFoundError, InsufficientDiskSpaceError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: scan failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        db.close()
        engine.close()
    print(f"Run {summary.run_id} {summary.status}: found={summary.total_found} processed={summary.processed} "
          f"reused={summary.reused} skipped={summary.skipped} errors={summary.errors} "
          f"in {summary.elapsed_seconds:.1f} s")
    return 1 if summary.errors else 0


def command_export_csv(args: argparse.Namespace) -> int:
    from database import Database, export_results_csv

    with Database() as db:
        count = export_results_csv(db, Path(args.out), label=args.label, precision=args.precision)
    print(f"Exported {count} rows to {args.out}")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from benchmark import export_benchmarks, parse_batch_sizes, run_benchmark
    from database import Database
    from scanner import discover_images

    try:
        batch_sizes = parse_batch_sizes(args.batch_sizes)
        paths = discover_images(Path(args.folder), recursive=True)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not paths:
        print("ERROR: no supported images found in the folder", file=sys.stderr)
        return 2
    loaded = load_engine(args.device, args.precision)
    if loaded is None:
        return 2
    engine, gpu_name, _revision = loaded
    stop_flag = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        stop_flag["requested"] = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        with Database() as db:
            results = run_benchmark(engine, paths, batch_sizes, gpu_name, image_limit=args.limit, db=db,
                                    should_stop=lambda: stop_flag["requested"])
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        engine.close()
    rows = [[str(r.batch_size), str(r.effective_batch_size), str(r.images), f"{r.total_seconds:.2f}",
             f"{r.images_per_second:.1f}", f"{r.ms_per_image:.1f}", f"{r.model_images_per_second:.1f}",
             f"{r.model_ms_per_image:.1f}", format_optional(
                 None if r.peak_memory_bytes is None else r.peak_memory_bytes / 2**20, "{:.0f} MiB")]
            for r in results]
    print_table(["batch", "effective", "images", "seconds", "img/s", "ms/img", "model img/s", "model ms/img",
                 "peak mem"], rows)
    if args.json:
        export_benchmarks(results, Path(args.json), "json")
        print(f"Wrote {args.json}")
    if args.csv:
        export_benchmarks(results, Path(args.csv), "csv")
        print(f"Wrote {args.csv}")
    return 0 if results else 1


def command_evaluate(args: argparse.Namespace) -> int:
    from database import Database
    from evaluation import LABELS, compute_metrics, export_metrics, import_folder_labels

    with Database() as db:
        if args.import_folder_labels:
            try:
                imported = import_folder_labels(db, Path(args.import_folder_labels))
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
            print(f"Imported {imported} folder labels")
        pairs = db.labeled_pairs(args.precision)
    metrics = compute_metrics((p["true_label"], p["predicted_label"]) for p in pairs)
    print(f"Precision mode: {args.precision}  labeled images with results: {metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
    print_table(["class", "precision", "recall", "f1", "support", "accuracy"],
                [[c.label, f"{c.precision:.4f}", f"{c.recall:.4f}", f"{c.f1:.4f}", str(c.support),
                  f"{c.accuracy:.4f}"] for c in metrics.per_class])
    print(f"Macro precision {metrics.macro_precision:.4f}  recall {metrics.macro_recall:.4f}  "
          f"f1 {metrics.macro_f1:.4f}  weighted f1 {metrics.weighted_f1:.4f}")
    print("Confusion matrix (rows = true, columns = predicted)")
    print_table(["true\\pred", *LABELS], [[label, *map(str, row)] for label, row in zip(LABELS, metrics.confusion)])
    if args.json:
        export_metrics(metrics, Path(args.json), "json")
        print(f"Wrote {args.json}")
    if args.csv:
        export_metrics(metrics, Path(args.csv), "csv")
        print(f"Wrote {args.csv}")
    return 0


def command_history(args: argparse.Namespace) -> int:
    from database import Database

    with Database() as db:
        runs = db.list_runs()
    print_table(["id", "started", "status", "folder", "device", "prec", "found", "proc", "reused", "skip", "err"],
                [[str(r["id"]), str(r["started_at"]), str(r["status"]), str(r["input_folder"]),
                  str(r["device"]), str(r["precision"]), str(r["total_found"]), str(r["processed"]),
                  str(r["reused"]), str(r["skipped"]), str(r["errors"])] for r in runs])
    return 0


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.load()
    parser = argparse.ArgumentParser(prog="blueeye", description="BlueEye Portable Tester")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="show PyTorch, GPUs and model status").set_defaults(handler=command_devices)

    def add_engine_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--precision", choices=PRECISIONS, default=settings.precision)
        p.add_argument("--device", default=settings.device, help="auto, cpu or cuda:N")

    scan = sub.add_parser("scan", help="classify images in a folder")
    scan.add_argument("folder")
    scan.add_argument("--no-recursive", action="store_true")
    scan.add_argument("--batch-size", type=int, default=settings.batch_size)
    scan.add_argument("--force", action="store_true")
    add_engine_options(scan)
    scan.set_defaults(handler=command_scan)

    export = sub.add_parser("export-csv", help="export stored results to CSV")
    export.add_argument("--out", default=str(RESULTS_CSV_PATH))
    export.add_argument("--label", choices=LABEL_CHOICES)
    export.add_argument("--precision", choices=PRECISIONS)
    export.set_defaults(handler=command_export_csv)

    bench = sub.add_parser("benchmark", help="measure throughput")
    bench.add_argument("folder")
    bench.add_argument("--batch-sizes", default=settings.benchmark_batch_sizes)
    bench.add_argument("--limit", type=int, default=settings.benchmark_image_limit)
    bench.add_argument("--json")
    bench.add_argument("--csv")
    add_engine_options(bench)
    bench.set_defaults(handler=command_benchmark)

    evaluate = sub.add_parser("evaluate", help="compute accuracy against labels")
    evaluate.add_argument("--precision", choices=PRECISIONS, default="fp32")
    evaluate.add_argument("--import-folder-labels", metavar="FOLDER")
    evaluate.add_argument("--json")
    evaluate.add_argument("--csv")
    evaluate.set_defaults(handler=command_evaluate)

    sub.add_parser("history", help="list previous runs").set_defaults(handler=command_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    from database import Database
    from logging_setup import configure_logging

    isolate_huggingface_environment()
    ensure_directories()
    configure_logging(console=False)
    with Database() as db:
        db.mark_interrupted_runs()
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
