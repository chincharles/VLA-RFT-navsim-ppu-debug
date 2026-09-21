#!/usr/bin/env python3
"""Prepare verified reward weights and a candidate VLM for the ClearML launcher.

Requires RFT_CACHE_ROOT and RFT_WORK. Only weights.json is written in the run
directory; reusable downloads live in the persistent cache. A supplied VLM_DIR
is used locally. Actual VLM compatibility is checked by check_vlm.py afterwards.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.request import Request, urlopen


VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
LPIPS_URL = (
    "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
    "master/lpips/weights/v0.1/vgg.pth"
)
DEFAULT_VLM_REPO = "VLA-Adapter/LIBERO-Object"
CHUNK_BYTES = 1024 * 1024


class WeightError(RuntimeError):
    """A safe-to-print error with no HTTP response or credential contents."""


def digest_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path, algorithm, expected, prefix=False):
    path = Path(path)
    if not path.is_file():
        raise WeightError(f"Expected a regular weight file: {path}")
    actual = digest_file(path, algorithm)
    valid = actual.startswith(expected) if prefix else actual == expected
    if not valid:
        raise WeightError(
            f"Hash mismatch for {path}; existing files are never replaced. "
            "Inspect this cache file before retrying."
        )
    return actual


def download_verified(url, target, algorithm, expected, prefix=False):
    """Caller holds the cache lock; publish without replacing any existing file."""
    target = Path(target)
    if target.exists() or target.is_symlink():
        return check_hash(target, algorithm, expected, prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as output:
                request = Request(url, headers={"User-Agent": "navsim-rft-weight-bootstrap/1"})
                with urlopen(request, timeout=60) as response:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            actual = check_hash(temporary, algorithm, expected, prefix)
            # Hard-link publication is atomic and fails if another process has
            # created target, unlike rename/replace. The temp is on the same FS.
            try:
                os.link(temporary, target)
            except FileExistsError:
                actual = check_hash(target, algorithm, expected, prefix)
            return actual
        except WeightError:
            raise
        except Exception as exc:
            if attempt == 2:
                raise WeightError(
                    f"Could not download {target.name} after 3 attempts "
                    f"({type(exc).__name__}); check outbound network and cache permissions."
                ) from None
            time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def prepare_vlm(cache_root):
    local = os.environ.get("VLM_DIR")
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_dir():
            raise WeightError(f"VLM_DIR is not an existing directory: {path}")
        return {
            "path": str(path),
            "repo": None,
            "revision": None,
            "source": "user-provided local checkpoint; provenance supplied by user",
            "status": "strict weight loading and real-image forward still required",
        }
    repo = os.environ.get("RFT_VLM_REPO", DEFAULT_VLM_REPO)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise WeightError("RFT_VLM_REPO must have the form owner/model, not a URL or credential.")
    requested_revision = os.environ.get("RFT_VLM_REVISION") or "main"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    try:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(repo_id=repo, revision=requested_revision).sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise WeightError("Hugging Face did not return an immutable 40-character model commit.")
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            cache_dir=str(Path(os.environ["HF_HOME"]).expanduser() / "hub"),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer.model"],
        )
    except WeightError:
        raise
    except Exception as exc:
        # HTTP exceptions may contain request details; retain the exception type
        # without echoing headers, auth tokens, or arbitrary response content.
        raise WeightError(
            f"VLM download failed ({type(exc).__name__}); check model access, network, "
            "RFT_VLM_REPO/RFT_VLM_REVISION, or supply a local VLM_DIR."
        ) from None
    path = Path(path).resolve()
    if not path.is_dir():
        raise WeightError("Hugging Face returned a missing snapshot directory.")
    return {
        "path": str(path),
        "repo": repo,
        "revision": revision,
        "source": (
            "public VLA-Adapter robot-finetuned initialization candidate; not VLA-RFT author weights"
            if repo == DEFAULT_VLM_REPO
            else "user-selected public/private initialization candidate; provenance not verified as VLA-RFT"
        ),
        "status": "strict weight loading and real-image forward still required",
    }


def main():
    for key in ("RFT_CACHE_ROOT", "RFT_WORK"):
        if not os.environ.get(key):
            raise WeightError(f"Set {key} before running this script.")
    cache_root = Path(os.environ["RFT_CACHE_ROOT"]).expanduser().resolve()
    work = Path(os.environ["RFT_WORK"]).expanduser().resolve()
    if not work.is_dir():
        raise WeightError(f"RFT_WORK must already be an independent run directory: {work}")
    report_path = work / "weights.json"
    if report_path.exists() or report_path.is_symlink():
        raise WeightError(f"Refusing to overwrite existing report: {report_path}")
    weight_root = cache_root / "weights"
    weight_root.mkdir(parents=True, exist_ok=True)
    with (weight_root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        vgg = weight_root / "vgg16-397923af.pth"
        lpips = weight_root / "vgg.pth"
        vgg_digest = download_verified(VGG_URL, vgg, "sha256", "397923af", prefix=True)
        lpips_digest = download_verified(
            LPIPS_URL, lpips, "md5", "d507d7349b931f0638a25a48a722f98a"
        )
        vlm = prepare_vlm(cache_root)
        report = {
            "vlm": vlm["path"],
            "vgg": str(vgg),
            "lpips": str(lpips),
            "vlm_source": vlm,
            "reward_sources": {
                "vgg": {"url": VGG_URL, "sha256": vgg_digest, "verified_prefix": "397923af"},
                "lpips": {"url": LPIPS_URL, "md5": lpips_digest},
            },
        }
        with report_path.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    print(f"Verified reward weights and VLM location recorded in {report_path}", flush=True)
    print("VLM remains a candidate until check_vlm.py passes.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WeightError as exc:
        print(f"Weight preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        print(
            f"Weight preparation failed ({type(exc).__name__}); inspect cache/run permissions and paths.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
