from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from settings import MODEL_DIR, isolate_huggingface_environment

REPO_ID = "pp1618/Blue-Eye"
REQUIRED_FILES = ("config.json", "model.safetensors")
MANIFEST_NAME = "manifest.json"
HASH_CHUNK = 8 * 1024 * 1024
DOWNLOAD_SPACE_MARGIN = 200 * 1024 * 1024


class ModelError(Exception):
    pass


@dataclass
class ModelStatus:
    present: bool
    message: str
    revision: str | None = None


def manifest_path() -> Path:
    return MODEL_DIR / MANIFEST_NAME


def read_manifest() -> dict | None:
    try:
        return json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_of(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(b"blob %d\0" % len(data))
    digest.update(data)
    return digest.hexdigest()


def check_model(full_hash: bool = False) -> ModelStatus:
    manifest = read_manifest()
    if manifest is None:
        missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
        if missing:
            return ModelStatus(False, f"Model not downloaded (missing: {', '.join(missing)}). Run Setup.bat.")
        return ModelStatus(False, "Model files exist but were never verified. Run Setup.bat to verify them.")
    for name in REQUIRED_FILES:
        entry = manifest["files"].get(name)
        path = MODEL_DIR / name
        if entry is None:
            return ModelStatus(False, f"Manifest has no entry for {name}. Run Setup.bat.")
        if not path.is_file():
            return ModelStatus(False, f"Missing model file: {name}. Run Setup.bat.")
        if path.stat().st_size != entry["size"]:
            return ModelStatus(False, f"Size mismatch for {name}: file is damaged or incomplete. Run Setup.bat.")
        if full_hash and not _hash_matches(path, entry):
            return ModelStatus(False, f"Checksum mismatch for {name}. Run Setup.bat to re-download it.")
    return ModelStatus(True, "Model ready", manifest.get("revision"))


def _hash_matches(path: Path, entry: dict) -> bool:
    if entry.get("sha256"):
        return sha256_of(path) == entry["sha256"]
    if entry.get("git_sha1"):
        return git_blob_sha1_of(path) == entry["git_sha1"]
    return False


def _resolve_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    print("A Hugging Face access token is required (the model is gated).")
    print(f"1. Open https://huggingface.co/{REPO_ID} and accept the access conditions.")
    print("2. Create a Read token at https://huggingface.co/settings/tokens")
    token = getpass.getpass("Paste the token (input is hidden, it is not saved): ").strip()
    return token or None


def _remote_file_info(token: str | None) -> tuple[str, dict]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(REPO_ID, files_metadata=True, token=token)
    except GatedRepoError as e:
        raise ModelError(
            f"Access to {REPO_ID} is denied. Accept the conditions at https://huggingface.co/{REPO_ID} "
            "with the same account the token belongs to."
        ) from e
    except RepositoryNotFoundError as e:
        raise ModelError(f"Repository {REPO_ID} not found or the token is invalid.") from e
    except HfHubHTTPError as e:
        raise ModelError(f"Hugging Face request failed: {e}") from e
    files = {}
    for sibling in info.siblings or []:
        if sibling.rfilename in REQUIRED_FILES:
            files[sibling.rfilename] = {
                "size": sibling.size,
                "sha256": sibling.lfs.sha256 if sibling.lfs else None,
                "git_sha1": sibling.blob_id if not sibling.lfs else None,
            }
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        raise ModelError(f"Repository does not contain: {', '.join(missing)}")
    return info.sha, files


def _check_free_space(files: dict) -> None:
    needed = sum(
        entry["size"] for name, entry in files.items()
        if not ((MODEL_DIR / name).is_file() and (MODEL_DIR / name).stat().st_size == entry["size"])
    )
    free = shutil.disk_usage(MODEL_DIR).free
    if needed + DOWNLOAD_SPACE_MARGIN > free:
        raise ModelError(
            f"Not enough disk space: need about {(needed + DOWNLOAD_SPACE_MARGIN) / 2**30:.2f} GiB, "
            f"free {free / 2**30:.2f} GiB."
        )


def _write_manifest(revision: str | None, files: dict, verified_against_hub: bool) -> None:
    manifest = {"repo_id": REPO_ID, "revision": revision, "verified_against_hub": verified_against_hub,
                "files": files}
    manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def adopt_local_files() -> ModelStatus:
    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise ModelError(f"Cannot import manually placed model: missing {', '.join(missing)} in {MODEL_DIR}")
    try:
        revision, remote = _remote_file_info(token=None)
    except ModelError as e:
        print(f"Official checksums unavailable ({e}). Recording local checksums instead.")
        revision, remote = None, None

    files = {}
    for name in REQUIRED_FILES:
        path = MODEL_DIR / name
        print(f"Checking {name}...")
        if remote is not None:
            entry = remote[name]
            if path.stat().st_size != entry["size"] or not _hash_matches(path, entry):
                raise ModelError(f"{name} does not match the official file on Hugging Face. Download it again.")
            files[name] = entry
        else:
            files[name] = {"size": path.stat().st_size, "sha256": sha256_of(path), "git_sha1": None}
    _write_manifest(revision, files, verified_against_hub=remote is not None)
    if remote is not None:
        print(f"Manually placed model matches the official files (revision {revision}).")
        return ModelStatus(True, "Model ready", revision)
    print("Manually placed model recorded; it was NOT compared with the official checksums.")
    return ModelStatus(True, "Model ready (not verified against Hugging Face)")


def download_model(force: bool = False) -> ModelStatus:
    os.environ["HF_HUB_OFFLINE"] = "0"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not force:
        status = check_model(full_hash=False)
        if status.present:
            print("Model already downloaded and verified. Skipping download.")
            return status
        if read_manifest() is None and all((MODEL_DIR / name).is_file() for name in REQUIRED_FILES):
            print("Found manually placed model files.")
            return adopt_local_files()

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    token = _resolve_token()
    revision, files = _remote_file_info(token)
    _check_free_space(files)

    for name, entry in files.items():
        path = MODEL_DIR / name
        if not force and path.is_file() and path.stat().st_size == entry["size"] and _hash_matches(path, entry):
            print(f"{name}: already present and valid")
            continue
        print(f"Downloading {name} ({entry['size'] / 2**20:.1f} MiB)...")
        try:
            hf_hub_download(REPO_ID, name, revision=revision, token=token, local_dir=MODEL_DIR,
                            force_download=force)
        except GatedRepoError as e:
            raise ModelError(f"Access to {REPO_ID} is denied. Accept the conditions on the model page.") from e
        except HfHubHTTPError as e:
            raise ModelError(f"Download of {name} failed: {e}. Run Setup.bat again to resume.") from e
        print(f"Verifying {name}...")
        if path.stat().st_size != entry["size"] or not _hash_matches(path, entry):
            path.unlink(missing_ok=True)
            raise ModelError(f"{name} failed checksum verification and was deleted. Run Setup.bat again.")

    _write_manifest(revision, files, verified_against_hub=True)
    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)
    print(f"Model ready (revision {revision}).")
    return ModelStatus(True, "Model ready", revision)


def main(argv: list[str] | None = None) -> int:
    isolate_huggingface_environment()
    parser = argparse.ArgumentParser(description="Blue-Eye model management")
    parser.add_argument("command", choices=["download", "verify", "status"])
    parser.add_argument("--force", action="store_true", help="re-download even if valid files exist")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download_model(force=args.force)
            return 0
        status = check_model(full_hash=args.command == "verify")
        print(status.message)
        return 0 if status.present else 1
    except ModelError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Run Setup.bat again to resume the download.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
