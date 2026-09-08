# Runtime snapshots

This directory contains explicitly exported copies of the project's local runtime.
The ZIP files are tracked with **Git LFS**. Each snapshot includes:

| Archive | Contents |
| --- | --- |
| `virtual-environment.zip` | The current `.venv/`, including installed Python packages and Windows binaries |
| `conversation-data.zip` | The current `data/`, including conversation journals, adapters, replay state, and checkpoint metadata |
| `model-cache.zip` | The current project-local `model_cache/` |

`manifest.json` records the export time, environment, file counts, archive sizes,
and SHA-256 checksums. During export, every archived file is compared with its
source. SQLite journals also receive an integrity check. Internal model-cache
file symlinks are stored as ordinary files so Windows can restore them without
symlink privileges.

These are snapshots of the files present at export time. The included model cache
contains the tiny test model; it does not contain every model named in the config
or models installed separately through Ollama.

## Download and verify

Clone this repository with Git LFS installed, or download the LFS content after cloning:

```shell
git lfs install
git lfs pull
```

Using a Python environment with TRIAD's dependencies installed:

```shell
python scripts/export_runtime_snapshot.py --verify runtime-snapshots/SNAPSHOT_ID
```

Replace `SNAPSHOT_ID` with the timestamp directory you downloaded. A plain GitHub
source ZIP may contain LFS pointer files instead of the binary archives, depending
on the repository's archive settings.

## Restore

Extract each archive into the **same empty destination folder**. The archives
already contain their `.venv/`, `data/`, and `model_cache/` top-level directories.
Copy the project's source files into that destination separately, or restore only
the selected runtime directories into a fresh clone.

The `.venv` is a backup of a particular Windows Python installation. Python virtual
environments contain absolute paths and depend on their original base Python;
copying this archive does **not** make it portable or self-contained. On another
machine, recreate `.venv` using the main README's installation instructions and
`requirements-tested.txt`. The recorded environment uses the CPU build of PyTorch.

## Export another snapshot

Stop TRIAD chat/training sessions, package installation, and model downloads, then run:

```shell
python scripts/export_runtime_snapshot.py
```

The script creates a new timestamp directory only after all archives pass
verification. It takes the existing per-user file locks while copying conversation
data and refuses files that change during export. Review the new snapshot, then
commit it and push using GitHub Desktop.

The live `.venv/`, `data/`, and `model_cache/` remain ignored. Exporting is an explicit
choice: subsequent conversations are included only when you create another snapshot.
An exported conversation archive contains the actual data, without anonymization.
Anyone who can read this repository can download the snapshots. Files outside the
three selected directories, such as `.env`, IDE caches, and Git repair backups, are
not part of this export.

Each new LFS archive version consumes additional LFS storage. See the
[GitHub LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
[LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs), and
[Python virtual environment portability](https://docs.python.org/3.12/library/venv.html#how-venvs-work).
