# BlueEye Portable Tester

A portable Windows application for testing the [Blue-Eye](https://github.com/prnshu-p/Blue-Eye) image classifier (safe / suggestive / explicit) locally on an NVIDIA GPU or the CPU. You can compare speed between computers and measure classification quality on your own labeled images.

The whole folder is self-contained. It holds Python, PyTorch with the CUDA runtime, the model, and all data. You can copy it to a USB drive or external SSD and run it on another Windows PC without installing anything except the NVIDIA driver.

## Contents

- [Requirements](#requirements)
- [Initial setup](#initial-setup)
- [Hugging Face access](#hugging-face-access)
- [Downloading the model](#downloading-the-model)
- [Launching](#launching)
- [Using the application](#using-the-application)
- [Moving to another computer](#moving-to-another-computer)
- [GPU requirements](#gpu-requirements)
- [Command line](#command-line)
- [Folder layout](#folder-layout)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11, 64-bit |
| Disk space | about 6.5 GB (runtime 4.8 GB, model 1.2 GB, data) |
| Internet | only for `Setup.bat` on the preparation computer |
| Admin rights | not required |
| GPU (optional) | NVIDIA GTX 16xx / RTX 20xx or newer with a recent driver; see [GPU requirements](#gpu-requirements) |

## Initial setup

On the preparation computer, which needs internet access:

1. Run `Setup.bat`. It:
   - downloads the official embeddable Python 3.12.10 into `runtime\python`;
   - installs pip and the pinned dependencies (PyTorch 2.11.0 with CUDA 12.8, transformers, PySide6, and others) into that folder;
   - downloads or verifies the Blue-Eye model in `model\`.
2. Run `Start.bat`.

`Setup.bat` is safe to run again. Steps that are already complete are skipped, and an interrupted model download resumes.

Nothing is installed system-wide. There are no registry entries and no PATH changes. Any global Python is neither used nor modified.

## Hugging Face access

The model repository [pp1618/Blue-Eye](https://huggingface.co/pp1618/Blue-Eye) is gated. Approval is automatic:

1. Sign in to Hugging Face, open the model page, and accept the conditions ("Agree and access").
2. For an automatic download, create a **Read** token at <https://huggingface.co/settings/tokens>.

The token is requested with hidden input during `Setup.bat` and is never saved to disk. You can also set the `HF_TOKEN` environment variable in the console before running `Setup.bat`.

## Downloading the model

There are two options.

- **Automatic.** `Setup.bat` asks for the token and downloads `config.json` and `model.safetensors` (1.2 GB) into `model\`. It shows progress, can resume, and checks free disk space.
- **Manual.** After accepting the conditions, download `config.json` and `model.safetensors` from the "Files and versions" tab in your browser. Put them into `model\` and run `Setup.bat`. No token is needed in this case.

In both cases the files are checked against the official SHA-256 values published by Hugging Face, and the result is recorded in `model\manifest.json`. If Hugging Face cannot be reached during a manual import, local checksums are recorded instead and the manifest shows `"verified_against_hub": false`.

At every start the application checks that the model files are present and have the expected size. A full checksum check is available on the Dashboard tab.

## Launching

Run `Start.bat`. If the runtime or the model is missing, it tells you to run `Setup.bat`.

Normal use is fully offline. Hugging Face offline mode is forced, and the Hugging Face cache is kept inside `runtime\hf_home`, never in the user profile.

## Using the application

| Tab | Purpose |
|---|---|
| Dashboard | Model status and full checksum check. PyTorch and CUDA versions. All detected GPUs, including incompatible ones and the reason. The current device. Totals per class. |
| Scan | Choose the folder and whether to scan subfolders, the batch size, device and precision. Start and stop. Shows progress, the current file and speed. |
| Results | Sortable and filterable table of all results. Export to CSV. Open a file or its folder. |
| Benchmark | Speed test with several batch sizes. Shows images per second, ms per image and peak GPU memory. Export to CSV or JSON. Speed only. |
| Evaluation | Label images by hand (keys 1/2/3, 0 to clear), or import labels from subfolder names. Shows accuracy, precision, recall, F1, per-class accuracy and the confusion matrix. Quality only. |

Behavior worth knowing:

- **Original files are never modified, moved or renamed.** Files are only opened for reading.
- **Results are stored in `data\results.db` (SQLite).** Each batch is committed in its own transaction, so an interrupted run never corrupts the database.
- **Resume and duplicate detection.**
  - A file that already has a valid result for the selected precision is skipped. Unchanged files are recognised by size and modification time, changed ones by SHA-256.
  - A file with the same content as an already classified file reuses that result without running the model.
  - Tick "Reprocess files that already have results" to force a new run.
- **Fast evaluation setup.** Put test images into subfolders named `safe`, `suggestive` and `explicit`, then use "Import labels from subfolder names". Manual labels always take priority over folder labels.
- **Ground-truth labels are stored by file hash.** They survive moving the folder between computers.
- **Benchmark runs never write to the results table.** Quality metrics never depend on benchmark runs.

## Moving to another computer

1. Finish `Setup.bat` on the preparation computer and check that `Start.bat` works.
2. Close the application.
3. Copy the **entire** `BlueEyePortable` folder to a USB drive or external SSD. NTFS or exFAT is recommended. No file is larger than 4 GB, so FAT32 also works, but it is slow with many small files.
4. On the target computer, run `Start.bat` directly from the drive, or copy the folder to a local disk first for faster startup.
5. The GPU is detected again at every start. Nothing from the preparation computer's GPU is assumed.

Requirements on the target computer:

- Windows 10 or 11, 64-bit.
- For GPU mode, an NVIDIA driver recent enough for CUDA 12.8 (see below). The driver is the only component that cannot be portable: it is part of the operating system and installing it requires admin rights.

Without a suitable driver or GPU, the application runs on the CPU.

Results from both computers are kept in the same database. Each result records the GPU and precision that produced it, and each benchmark run is stored with its GPU name, so speed comparisons can be exported side by side.

## GPU requirements

The bundled PyTorch build (2.11.0+cu128) contains GPU code for compute capabilities **sm_75, sm_80, sm_86, sm_90, sm_100, sm_120**:

| Supported | Examples |
|---|---|
| Turing (sm_75) | GTX 1650/1660, RTX 2060–2080 |
| Ampere (sm_80/86) | RTX 3050–3090, A-series |
| Ada (sm_89, runs the sm_86 code) | RTX 4060–4090 |
| Hopper / Blackwell (sm_90/100/120) | H100, RTX 5070–5090 |

**Not supported:** Pascal and older, for example GTX 10xx, GTX 9xx and GTX 7xx. These GPUs are listed as incompatible and the application falls back to the CPU.

**NVIDIA driver:** version 570 or newer is recommended. It is required for RTX 50 series cards. Thanks to CUDA minor-version compatibility, older drivers (528.33 or newer) may work on older GPUs, but that has not been tested. The CUDA Toolkit is **not** required, because PyTorch bundles the CUDA runtime.

**Precision:**

- `fp32` is the official default. It works everywhere.
- `bf16` is the official alternative mode. Only the backbone runs in bfloat16; the head and the softmax stay float32. It is offered only on GPUs with native bfloat16 support (compute capability 8.0 or higher, RTX 30xx and newer) and on CPUs with native bf16 instructions. Results may differ slightly from fp32, so results are stored separately for each precision.
- The official implementation has no `fp16` mode, so this application does not provide one.

**Memory:** with fp32, peak GPU memory is about 1.2 GB at batch size 1 and 1.5 GB at batch size 8, so a 6 GB card is enough. The default batch size is 4. If a batch runs out of GPU memory, it is split in half and retried automatically, and the smaller size is used for the rest of the run. Results do not change.

**Multiple GPUs:** the compatible GPU with the most memory is selected automatically. You can choose another one, or the CPU, on the Scan and Benchmark tabs.

## Command line

Every function is also available headless:

```bat
runtime\python\python.exe app\cli.py devices
runtime\python\python.exe app\cli.py scan input --batch-size 4 --precision fp32
runtime\python\python.exe app\cli.py export-csv --out output\results.csv
runtime\python\python.exe app\cli.py benchmark input --batch-sizes 1,2,4,8 --limit 64 --json output\bench.json
runtime\python\python.exe app\cli.py evaluate --import-folder-labels input --json output\metrics.json
runtime\python\python.exe app\cli.py history
runtime\python\python.exe app\model_manager.py verify
```

Run the tests with:

```bat
runtime\python\python.exe -m unittest discover -s tests -t .
```

## Folder layout

```
BlueEyePortable/
  Start.bat              launch the application
  Setup.bat              one-time preparation (internet required)
  README.md
  setup/                 bootstrap script and pinned requirements
  app/
    main.py              GUI entry point
    cli.py               command-line interface
    classifier.py        inference engine (calls the official code)
    gpu.py               GPU detection and compatibility checks
    scanner.py           folder scanning, resume, duplicate detection
    database.py          SQLite storage and CSV export
    benchmark.py         speed measurements
    evaluation.py        quality metrics
    model_manager.py     model download and verification
    settings.py          paths (all relative to this folder) and settings
    logging_setup.py     log files
    ui/                  PySide6 interface
    official/            unmodified official Blue-Eye code (commit aad35f4) and its license
  model/                 config.json, model.safetensors, manifest.json
  runtime/python/        embeddable Python with all packages
  input/                 put test images here (optional)
  output/
    results.csv          default CSV export
    logs/                app.log, inference.log, errors.log
  data/
    results.db           results, runs, labels, benchmarks
    settings.json        UI settings
  tests/                 unit and UI smoke tests
```

Why the embeddable Python and not a virtual environment: a `venv` stores absolute paths to the base interpreter and breaks when moved. The embeddable distribution is a plain folder. Its `python312._pth` file uses only relative paths, and it ignores `PYTHONPATH` and the user site-packages, so a global Python cannot interfere.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Portable Python runtime not found" | `Setup.bat` was never run, or the folder was copied only partially. Copy the entire folder again. |
| "Model not downloaded" or a size mismatch | Run `Setup.bat`. To repair, delete the damaged files in `model\` and run it again. |
| Access denied / gated repository | Accept the conditions on the model page with the same account the token belongs to. |
| GPU listed as "not supported by this PyTorch build" | The GPU is older than Turing (for example GTX 10xx or older). The CPU is used instead. |
| CUDA not available on a supported GPU | Install or update the NVIDIA driver (570 or newer recommended). Check with `nvidia-smi`. The application offers to switch to the CPU. |
| `CUDA out of memory even with batch size 1` | Another program is using GPU memory. Close it, or use the CPU. |
| bf16 option disabled | The GPU or CPU has no native bfloat16 support (for example GTX 16xx or RTX 20xx). Use fp32. |
| Slow first start from a USB stick | About 5 GB of libraries and weights are read from the drive. Use a USB 3 SSD, or copy the folder to a local disk. |
| Anything else | See `output\logs\errors.log` and `app.log`. |

## Known limitations

- **Center crop.** Following the official preprocessing, images are resized so the short side is 537 px, then center-cropped to 512×512. On wide or tall images the edges are not seen by the model.
- **No EXIF rotation.** EXIF orientation is not applied (official behavior, matching training). Phone photos may be classified while rotated.
- **First frame only.** Animated WEBP and multi-page TIFF files are classified by their first frame.
- **Per-image time.** "Processing time" per image is the image's own decode and preprocessing time plus its share of the batch's model time. Images inside a batch are not timed separately.
- **Very large images.** Images above Pillow's decompression-bomb limit (about 179 megapixels) are rejected as errors.
- **NVIDIA only.** AMD and Intel GPUs are not supported; such computers run on the CPU. Apple MPS does not apply on Windows.
- **What was tested.** All tests ran on one Windows 10 PC:
  - Inference on a GTX 16xx (sm_75, fp32) and on an 8-core CPU. Results matched the official `classify()` exactly. An unsupported pre-Turing GPU (sm_50) was correctly reported as incompatible.
  - CPU fallback with all GPUs hidden.
  - Relocation: the folder was mounted under different drive letters with `subst` and launched via `Start.bat`. Results stored from one location were recognised from the others.
  - Not yet done: a copy to a second physical computer. bf16 and the RTX 30/40/50 series have not been run on real hardware; their support is inferred from the PyTorch build's architecture list.
- **License.** The model and the official code are released under the DINOv3 License (see `app\official\LICENSE`). The Hugging Face access conditions also forbid using the model to monitor or make decisions about individual people.
