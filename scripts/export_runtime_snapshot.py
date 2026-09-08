"""Export the current local runtime as verified, explicitly shared ZIP archives.

Stop model downloads, package installs, and TRIAD sessions before running this script.
Only .venv/, data/, and model_cache/ are included; credentials elsewhere are not read.
"""

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    ".venv": "virtual-environment.zip",
    "data": "conversation-data.zip",
    "model_cache": "model-cache.zip",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(source):
    """Materialize internal file symlinks; never follow links outside the source."""
    result = {}
    for directory, folders, files in os.walk(source, followlinks=False):
        base = Path(directory)
        for name in folders:
            path = base / name
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"Directory links must be resolved before export: {path}")
        for name in files:
            path = base / name
            if not path.resolve(strict=True).is_relative_to(source):
                raise ValueError(f"File points outside its source directory: {path}")
            stat = path.stat()
            result[path] = (stat.st_size, stat.st_mtime_ns)
    return dict(sorted(result.items()))


def archive_source(source, target):
    with ExitStack() as locks:
        if source.name == "data":
            for path in sorted(source.rglob("user.lock")):
                locks.enter_context(FileLock(path, timeout=10))
        before = inventory(source)
        hashes = {}
        with zipfile.ZipFile(
            target,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            strict_timestamps=False,
        ) as archive:
            for path in before:
                name = path.relative_to(ROOT).as_posix()
                archive.write(path, name)
                hashes[name] = digest(path)
        if inventory(source) != before:
            raise RuntimeError(f"{source.name} changed during export; stop writers and retry.")
        with zipfile.ZipFile(target) as archive:
            if set(archive.namelist()) != set(hashes):
                raise RuntimeError(f"Archive file list mismatch: {target}")
            for name, expected in hashes.items():
                with archive.open(name) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Source/archive content mismatch: {name}")
        if inventory(source) != before:
            raise RuntimeError(f"{source.name} changed during verification; retry when idle.")
    size = target.stat().st_size
    if size >= 2_000_000_000:
        raise ValueError("Archive exceeds the conservative 2 GB Git LFS file limit.")
    return {
        "source": source.name,
        "archive": target.name,
        "files": len(before),
        "source_bytes": sum(size for size, _ in before.values()),
        "archive_bytes": size,
        "sha256": digest(target),
        "source_contents_verified": True,
        "file_symlinks": "materialized as regular files",
    }


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["archives"]:
        name = record["archive"]
        if Path(name).name != name or name not in SOURCES.values():
            raise ValueError("Unexpected archive filename in manifest.")
        path = directory / name
        if path.stat().st_size != record["archive_bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist()) != record["files"] or archive.testzip() is not None:
                raise ValueError(f"Archive contents are invalid: {name}")
            for member in archive.namelist():
                parts = member.split("/")
                if parts[0] != record["source"] or any(p in ("", ".", "..") for p in parts):
                    raise ValueError(f"Unexpected archive member: {member}")
                if member.endswith("/journal.sqlite3"):
                    with tempfile.TemporaryDirectory(prefix="triad-db-check-") as temporary:
                        db_path = Path(temporary) / "journal.sqlite3"
                        db_path.write_bytes(archive.read(member))
                        with closing(sqlite3.connect(db_path)) as database:
                            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                                raise ValueError(f"Database integrity check failed: {member}")
        print(f"Verified {name}: {record['files']} files", flush=True)
    return manifest


def export():
    timestamp = datetime.now(timezone.utc)
    destination = ROOT / "runtime-snapshots" / timestamp.strftime("%Y%m%dT%H%M%SZ")
    if destination.exists():
        raise FileExistsError(destination)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-export-", dir=ROOT / ".cache") as temporary:
        staging = Path(temporary)
        records = []
        for source_name, archive_name in SOURCES.items():
            source = ROOT / source_name
            if not source.is_dir():
                raise FileNotFoundError(source)
            print(f"Exporting {source_name} ...", flush=True)
            records.append(archive_source(source, staging / archive_name))
        manifest = {
            "format_version": 1,
            "created_at": timestamp.isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "git_parent_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "scope": "Current project .venv, data, and model_cache directories only.",
            "portability": "Windows environment backup; recreate the venv on other machines.",
            "archives": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        verify(staging)
        destination.parent.mkdir(exist_ok=True)
        os.replace(staging, destination)
    print(f"Snapshot ready: {destination.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, metavar="SNAPSHOT_DIRECTORY")
    arguments = parser.parse_args()
    if arguments.verify is not None:
        verify(arguments.verify.resolve())
    else:
        export()
