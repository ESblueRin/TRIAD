from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from triad.engine import TriadEngine
from triad.schemas import Interaction
from triad.storage import CorruptCheckpoint
from triad.student import delta_norm
from triad.tiny import make_tiny_student


def triple():
    return Interaction(user_query="Name?", student_response="Seongjun", user_followup="Juho")


@pytest.mark.parametrize("failure", ["nan_loss", "inf_gradient", "inf_weights", "oom"])
def test_numerical_failure_rolls_back_and_logs(engine, monkeypatch, failure):
    engine.learn("numeric", triple())
    before = engine.student.adapter_state()
    old_updates = engine.store("numeric").latest()[0].updates
    if failure == "nan_loss":
        monkeypatch.setattr(
            engine.student.model,
            "forward",
            lambda **_: SimpleNamespace(
                loss=torch.tensor(float("nan"), requires_grad=True),
            ),
        )
    elif failure == "inf_gradient":
        param = next(p for p in engine.student.model.parameters() if p.requires_grad)
        handle = param.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    else:
        original_step = torch.optim.AdamW.step

        def broken_step(optimizer, *args, **kwargs):
            original_step(optimizer, *args, **kwargs)
            if failure == "oom":
                raise torch.OutOfMemoryError("simulated CUDA OOM after an actual step")
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("inf"))

        monkeypatch.setattr(torch.optim.AdamW, "step", broken_step)
    event = engine.learn("numeric", triple())
    if failure == "inf_gradient":
        handle.remove()
    assert not event.training_performed and event.training.error
    assert delta_norm(engine.student.adapter_state(), before) == 0
    state, path, _ = engine.store("numeric").latest()
    assert state.updates == old_updates and state.interactions == 2
    assert delta_norm(engine.student.read_adapter(path), before) == 0
    assert all(torch.isfinite(p).all() for p in engine.student.adapter_state().values())
    assert len(engine.store("numeric").events()) == 2


def test_write_failure_keeps_previous_durable_and_memory_state(engine, monkeypatch):
    engine.learn("disk", triple())
    store = engine.store("disk")
    before = engine.student.adapter_state()
    previous_version = store.latest()[2]

    def interrupted_save(path):
        (path / "adapter_model.safetensors").write_bytes(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr(engine.student, "write_adapter", interrupted_save)
    with pytest.raises(OSError, match="disk full"):
        engine.learn("disk", triple())
    assert store.latest()[2] == previous_version
    assert len(store.events()) == 1
    assert delta_norm(engine.student.adapter_state(), before) == 0
    assert not list(store.snapshots.glob("tmp-*"))


def test_database_failure_cannot_publish_adapter_or_training_event(engine, monkeypatch):
    engine.learn("db", triple())
    store = engine.store("db")
    before = engine.student.adapter_state()
    previous = store.latest()[2]

    def fail(*args, **kwargs):
        raise RuntimeError("simulated journal failure")

    monkeypatch.setattr(store, "_journal", fail)
    with pytest.raises(RuntimeError, match="journal failure"):
        engine.learn("db", triple())
    assert store.latest()[2] == previous
    assert len(store.events()) == 1
    assert delta_norm(engine.student.adapter_state(), before) == 0


def test_corrupt_current_recovers_previous_checkpoint(engine):
    engine.learn("corrupt", triple())
    before = engine.student.adapter_state()
    engine.learn("corrupt", triple())
    store = engine.store("corrupt")
    _, current_path, version = store.latest()
    (current_path / "adapter_model.safetensors").write_bytes(b"broken checkpoint")
    with pytest.raises(CorruptCheckpoint):
        store.read(version)
    with pytest.warns(UserWarning, match="Recovered"):
        engine.answer("corrupt", "Name?")
    assert delta_norm(engine.student.adapter_state(), before) == 0
    assert store.events("recovery")[0]["from_version"] == version


def test_checkpoint_rollback_restores_replay_and_monotonic_versions(engine):
    engine.learn("rollback", triple())
    saved, _ = engine.checkpoint("rollback")
    before = engine.student.adapter_state()
    engine.learn("rollback", triple())
    engine.learn("rollback", triple())
    assert delta_norm(engine.student.adapter_state(), before) > 0
    restored = engine.rollback("rollback", saved)
    assert restored > saved
    assert delta_norm(engine.student.adapter_state(), before) == 0
    state, _, _ = engine.store("rollback").latest()
    assert state.updates == 1 and len(state.replay) == 1
    assert len(engine.store("rollback").events()) == 3  # audit history is never rolled back


def test_sparse_retention_and_initial_checkpoint(engine, config):
    config.storage.checkpoint_every_n_updates = 3
    for _ in range(8):
        engine.learn("sparse", triple())
    store = engine.store("sparse")
    retained = [row for row in store.checkpoints() if row["retained"]]
    assert {row["updates"] for row in retained} == {0, 3, 6, 7, 8}
    assert len(list(store.snapshots.iterdir())) == len(retained)
    engine.rollback("sparse", 0)
    assert delta_norm(engine.student.adapter_state(), engine.student.initial_adapter) == 0


def test_incompatible_config_is_not_silently_loaded(engine, config, teacher):
    engine.learn("signature", triple())
    changed = config.model_copy(deep=True)
    changed.lora.rank = 8
    other = TriadEngine(changed, make_tiny_student(changed), teacher)
    with pytest.raises(ValueError, match="fingerprint"):
        other.answer("signature", "Name?")


def test_multiple_engine_writers_preserve_all_events(config, teacher):
    engines = [TriadEngine(config, make_tiny_student(config), teacher) for _ in range(2)]

    def learn(engine):
        return [engine.learn("shared", triple()).training_performed for _ in range(2)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(learn, engines))
    assert all(all(result) for result in results)
    assert len(engines[0].store("shared").events()) == 4
    assert engines[0].store("shared").latest()[0].updates == 4


def test_user_id_cannot_escape_storage(config):
    from triad.storage import UserStore

    store = UserStore(config.storage, "../../another/person")
    assert store.directory.parent == config.storage.root.resolve() / "users"
    assert len(store.directory.name) == 64
