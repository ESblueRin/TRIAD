"""Transactional metadata plus immutable, checksum-verified PEFT snapshots.

The snapshot is flushed and renamed before SQLite commits the new head and event.
A crash before that commit leaves only an unreferenced directory, never a partial head.
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from filelock import FileLock

from .config import StorageConfig
from .schemas import TrainingEvent, UserState, utc_now


class CorruptCheckpoint(RuntimeError):
    pass


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def write_json(path: Path, value):
    """Atomic replacement, including for exported reports and event streams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json_text(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class UserStore:
    def __init__(self, config: StorageConfig, user_id: str):
        if not user_id.strip() or len(user_id) > 256:
            raise ValueError("user_id must contain 1-256 characters")
        self.config = config
        self.user_id = user_id
        self.directory = (
            config.root.resolve() / "users" / hashlib.sha256(user_id.encode()).hexdigest()
        )
        self.snapshots = self.directory / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.directory / "user.lock", timeout=config.lock_timeout_seconds)
        self.db_path = self.directory / "journal.sqlite3"
        with self.lock, self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    version INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,
                    manifest_sha256 TEXT NOT NULL, updates INTEGER NOT NULL,
                    pinned INTEGER NOT NULL, retained INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS head (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER);
                CREATE TABLE IF NOT EXISTS journal (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=self.config.lock_timeout_seconds)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _row(self, version: int):
        with self.connection() as db:
            row = db.execute("SELECT * FROM snapshots WHERE version=?", (version,)).fetchone()
        if row is None or not row["retained"]:
            raise FileNotFoundError(f"Checkpoint version {version} is not retained")
        return row

    def read(self, version: int) -> tuple[UserState, Path]:
        row = self._row(version)
        path = self.snapshots / row["path"]
        try:
            if path.resolve().parent != self.snapshots.resolve():
                raise ValueError("Invalid checkpoint directory")
            manifest_path = path / "manifest.json"
            if checksum(manifest_path) != row["manifest_sha256"]:
                raise ValueError("Manifest checksum mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "adapter_model.safetensors",
                "adapter_config.json",
                "state.json",
                "metadata.json",
            }
            if set(manifest["files"]) != expected:
                raise ValueError("Incomplete checkpoint manifest")
            for name, digest in manifest["files"].items():
                if checksum(path / name) != digest:
                    raise ValueError(f"Checksum mismatch: {name}")
            state = UserState.model_validate_json((path / "state.json").read_text(encoding="utf-8"))
            if state.user_id != self.user_id:
                raise ValueError("Checkpoint belongs to another user")
            return state, path
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CorruptCheckpoint(f"Invalid checkpoint v{version}: {exc}") from exc

    def latest(self) -> tuple[UserState, Path, int] | None:
        """Caller holds the user lock. Automatically recover the newest valid retained state."""
        with self.connection() as db:
            head = db.execute("SELECT version FROM head WHERE id=1").fetchone()
            if head is None:
                return None
            versions = [
                r[0]
                for r in db.execute(
                    "SELECT version FROM snapshots WHERE retained=1 AND version<=? ORDER BY version DESC",
                    (head[0],),
                )
            ]
        failures = []
        for version in versions:
            try:
                state, path = self.read(version)
            except (CorruptCheckpoint, FileNotFoundError) as exc:
                failures.append(str(exc))
                continue
            if failures:
                with self.connection() as db:
                    db.execute("UPDATE head SET version=? WHERE id=1", (version,))
                    self._journal(
                        db,
                        "recovery",
                        {
                            "from_version": head[0],
                            "to_version": version,
                            "errors": failures,
                        },
                    )
                warnings.warn(
                    f"Recovered {self.user_id!r} from v{head[0]} to v{version}", stacklevel=2
                )
            return state, path, version
        raise CorruptCheckpoint("No valid retained checkpoint; " + "; ".join(failures))

    @staticmethod
    def _journal(db, kind: str, payload: dict, event_id: str | None = None):
        db.execute(
            "INSERT INTO journal(event_id,timestamp,kind,payload) VALUES(?,?,?,?)",
            (event_id or uuid.uuid4().hex, utc_now(), kind, json_text(payload)),
        )

    def commit(
        self,
        state: UserState,
        write_adapter: Callable[[Path], None],
        metadata: dict,
        event: TrainingEvent | None = None,
        note: dict | None = None,
        pin: bool = False,
    ) -> tuple[int, Path]:
        """Caller holds lock; exceptions leave SQLite head/event and old snapshot unchanged."""
        with self.connection() as db:
            version = db.execute("SELECT COALESCE(MAX(version),-1)+1 FROM snapshots").fetchone()[0]
        identifier = f"v{version:08d}-{uuid.uuid4().hex[:12]}"
        staging = self.snapshots / ("tmp-" + uuid.uuid4().hex)
        destination = self.snapshots / identifier
        try:
            staging.mkdir()
            write_adapter(staging)
            write_json(staging / "state.json", state.model_dump(mode="json"))
            write_json(staging / "metadata.json", metadata)
            names = [
                "adapter_model.safetensors",
                "adapter_config.json",
                "state.json",
                "metadata.json",
            ]
            for name in names:
                with (staging / name).open("rb+") as handle:
                    os.fsync(handle.fileno())
            write_json(
                staging / "manifest.json", {"files": {n: checksum(staging / n) for n in names}}
            )
            manifest_sha256 = checksum(staging / "manifest.json")
            os.replace(staging, destination)
            pinned = (
                pin
                or version == 0
                or (
                    event is not None
                    and event.training_performed
                    and state.updates % self.config.checkpoint_every_n_updates == 0
                )
            )
            if event is not None:
                event.adapter_checkpoint = str(destination)
                event.adapter_version = version
            with self.connection() as db:
                db.execute(
                    "INSERT INTO snapshots(version,path,manifest_sha256,updates,pinned,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (version, identifier, manifest_sha256, state.updates, int(pinned), utc_now()),
                )
                db.execute("INSERT OR REPLACE INTO head(id,version) VALUES(1,?)", (version,))
                if event is not None:
                    self._journal(db, "training", event.model_dump(mode="json"), event.event_id)
                if note is not None:
                    self._journal(db, note.get("kind", "state"), note)
        finally:
            if staging.exists():
                self._remove_snapshot(staging)
        # Maintenance cannot turn an already committed update into a reported failure.
        try:
            self._prune()
        except Exception as exc:
            warnings.warn(f"Snapshot cleanup deferred: {exc}", stacklevel=2)
        return version, destination

    def _remove_snapshot(self, path: Path):
        if path.resolve().parent != self.snapshots.resolve() or not re.fullmatch(
            r"(?:v\d+-[0-9a-f]{12}|tmp-[0-9a-f]{32})",
            path.name,
        ):
            raise ValueError("Refusing to delete outside the snapshot store")
        shutil.rmtree(path)

    def _prune(self):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM snapshots ORDER BY version DESC").fetchall()
            # Keep initial, scheduled/manual pins, current head and its previous state.
            head = db.execute("SELECT version FROM head WHERE id=1").fetchone()[0]
            recent = [r["version"] for r in rows if r["version"] <= head and r["retained"]][:2]
            keep = {r["path"] for r in rows if r["pinned"] or r["version"] in recent}
            for row in rows:
                if row["path"] not in keep:
                    db.execute("UPDATE snapshots SET retained=0 WHERE version=?", (row["version"],))
        for path in self.snapshots.iterdir():
            if path.is_dir() and path.name not in keep:
                self._remove_snapshot(path)

    def checkpoints(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT version,updates,pinned,retained,created_at,path FROM snapshots ORDER BY version",
                )
            ]

    def events(self, kind: str | None = "training") -> list[dict]:
        with self.connection() as db:
            if kind is None:
                rows = db.execute("SELECT * FROM journal ORDER BY sequence").fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM journal WHERE kind=? ORDER BY sequence", (kind,)
                ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "kind": row["kind"],
                **json.loads(row["payload"]),
            }
            for row in rows
        ]
