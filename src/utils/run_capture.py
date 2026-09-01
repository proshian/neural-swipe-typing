"""Capture the git, data, and environment state a training run was launched from."""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import lightning


logger = logging.getLogger(__name__)

METADATA_FILENAME = "run_metadata.json"
PATCH_FILENAME = "dirty.patch"

# Untracked files larger than this are named in the metadata but left out of the
# patch, so that a stray dataset or checkpoint cannot bloat every run directory.
MAX_PATCH_FILE_BYTES = 1_000_000

# These differ by definition between a run and its own resume, so they are left
# out of the config fingerprint - otherwise every resume would look divergent.
FINGERPRINT_EXCLUDED_CONFIG_KEYS = ("path_to_continue_checkpoint", "allow_divergent_resume")


def _git(args: list[str], cwd: Path, ok_codes: tuple[int, ...] = (0,)) -> bytes:
    """Run git and return raw stdout.

    Output stays bytes: git has already decided how to encode paths and whether
    to normalize newlines, and text=True would redo both on top of that.

    Arguments:
    ----------
    args: list[str]
        Arguments for git, without the leading "git".
    cwd: Path
        Directory to run git in; must be inside the repository.
    ok_codes: tuple[int, ...]
        Exit codes to treat as success. Widen it for subcommands that report
        through the exit status: `git diff --no-index` exits 1 when files differ.

    Raises:
    -------
    RuntimeError
        If git is not installed/found, or exits with a code not in ok_codes.
    """
    try:
        # -c sets config for this call only. quotepath=off keeps non-ASCII paths
        # readable and usable, instead of escaping them ("\321\202...").
        completed = subprocess.run(["git", "-c", "core.quotepath=off", *args],
                                   cwd=str(cwd), capture_output=True)
    except OSError as e:
        raise RuntimeError(f"Could not run git in {cwd}: {e}") from e

    if completed.returncode not in ok_codes:
        raise RuntimeError(
            f"git {' '.join(args)} exited with {completed.returncode} in {cwd}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}")

    return completed.stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_referenced_files(node, repo_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (unresolved path, resolved path) for every config value naming a file.

    Walking the config instead of listing known keys means new data paths are
    covered automatically as configs grow.
    """
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_referenced_files(value, repo_root)
        return
    if isinstance(node, list):
        for item in node:
            yield from _iter_referenced_files(item, repo_root)
        return
    if not isinstance(node, str):
        return

    path = Path(node)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        return
    resolved = path.resolve()

    yield node, resolved


def _find_referenced_files(resolved_cfg: dict, repo_root: Path) -> list[tuple[str, Path]]:
    """
    Returns (unresolved path, resolved path) for every existing file named in the
    config, each file listed once even when several keys point at it.

    The unresolved path is the string as written in the config, e.g.
    "./data/data_preprocessed/voc.txt". The resolved path is the file it
    actually refers to: absolute, and with symlinks followed.
    """
    seen: set[Path] = set()
    found: list[tuple[str, Path]] = []
    for unresolved_path, resolved_path in _iter_referenced_files(resolved_cfg, repo_root):
        if resolved_path not in seen:
            seen.add(resolved_path)
            found.append((unresolved_path, resolved_path))
    return found


def _hash_data_files(files: list[tuple[str, Path]]) -> list[dict]:
    """Size and sha256 of each file. Reading a multi-GB dataset takes seconds."""
    return [{"unresolved_path": unresolved_path,
             "resolved_path": str(resolved_path),
             "size": resolved_path.stat().st_size,
             "sha256": _sha256_file(resolved_path)}
            for unresolved_path, resolved_path in files]


def _capture_uncommitted(repo_root: Path) -> tuple[bytes | None, list[str]]:
    """Return (patch bytes, untracked files left out of the patch)."""
    patch = _git(["diff", "--binary", "HEAD"], repo_root)
    skipped: list[str] = []

    # `git diff HEAD` only covers tracked files, so untracked ones are diffed
    # against /dev/null individually and appended to the same patch.
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], repo_root)
    # -z ends every name with a NUL, so the last piece of the split is empty.
    untracked_names = [name for name in untracked.decode("utf-8").split("\0") if name]

    for name in untracked_names:
        if (repo_root / name).stat().st_size > MAX_PATCH_FILE_BYTES:
            skipped.append(name)
            continue
        # --no-index answers through the exit code. Against /dev/null it is
        # always 1: the diff shows the file being created.
        patch += _git(["diff", "--binary", "--no-index", "--", "/dev/null", name],
                      repo_root, ok_codes=(1,))

    return (patch or None), skipped


def _capture_git(repo_root: Path) -> tuple[dict, bytes | None]:
    """Capture the git metadata as a dict and any uncommitted work as bytes.

    Returns:
    --------
    dict
        Git metadata: commit, is_dirty, status_porcelain, skipped_untracked_files.
    bytes | None
        Patch holding the uncommitted work, or None when the tree is clean.
    """
    commit = _git(["rev-parse", "HEAD"], repo_root).decode("ascii").strip()
    status = _git(["status", "--porcelain"], repo_root).decode("utf-8")
    is_dirty = bool(status.strip())

    patch: bytes | None = None
    skipped: list[str] = []
    if is_dirty:
        patch, skipped = _capture_uncommitted(repo_root)

    info = {
        "commit": commit,
        "is_dirty": is_dirty,
        # Stored unchanged: even when a file's contents are too big to capture,
        # the listing still records that the tree differed from the commit.
        "status_porcelain": status,
        "skipped_untracked_files": skipped,
    }
    
    return info, patch


def _config_fingerprint(resolved_cfg: dict[str, Any]) -> str:
    comparable = {k: v for k, v in resolved_cfg.items()
                  if k not in FINGERPRINT_EXCLUDED_CONFIG_KEYS}
    payload = json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)
    return _sha256_bytes(payload.encode("utf-8"))


def collect_run_state(resolved_cfg: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Gather everything needed to describe, and later reproduce, this launch.

    Returns a dict with the metadata and the (possibly None) patch bytes, so the
    caller decides where and whether to write them.
    """
    git_info, patch = _capture_git(repo_root)

    data_files = _hash_data_files(_find_referenced_files(resolved_cfg, repo_root))

    metadata = {
        "experiment_name": resolved_cfg.get("experiment_name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "available_gpus": [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())] or None,
            "lightning": lightning.__version__,
        },
        "git": git_info,
        "data_files": data_files,
        "fingerprint": {
            "commit": git_info.get("commit"),
            "patch": _sha256_bytes(patch) if patch else None,
            "config": _config_fingerprint(resolved_cfg),
        },
    }
    return {"metadata": metadata, "patch": patch}


def save_initial_run_state(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Write metadata and patch for a newly started run."""
    metadata = {**state["metadata"], "resumes": []}
    if state["patch"]:
        # Binary write: the patch must survive verbatim to stay applicable.
        (run_dir / PATCH_FILENAME).write_bytes(state["patch"])
    with open(run_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def append_resume_record(run_dir: Path, state: dict[str, Any],
                         divergent_parts: list[str]) -> int:
    """Record a resume against an existing run. Returns the 1-based resume index.

    The original launch's metadata is left untouched, so a run directory always
    tells the truth about how each stretch of training was produced.
    """
    with open(run_dir / METADATA_FILENAME, encoding="utf-8") as f:
        metadata = json.load(f)
    resumes = metadata.setdefault("resumes", [])
    index = len(resumes) + 1

    new = state["metadata"]
    entry = {
        "timestamp": new["timestamp"],
        "argv": new["argv"],
        "environment": new["environment"],
        "fingerprint": new["fingerprint"],
        "divergent_parts": divergent_parts,
    }
    # Record git only when it changed. Otherwise it would just repeat the
    # original launch's block, which the matching fingerprint already vouches for.
    if "commit" in divergent_parts or "patch" in divergent_parts:
        entry["git"] = new["git"]
    resumes.append(entry)

    if divergent_parts and state["patch"]:
        (run_dir / f"dirty.resume-{index}.patch").write_bytes(state["patch"])
    with open(run_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return index
