import pytest
import torch

from triad.engine import TriadEngine
from triad.schemas import TeacherDecision
from triad.tiny import make_tiny_student, tiny_config


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    torch.set_num_threads(1)


class FixedTeacher:
    def __init__(self, **kwargs):
        self.decision = TeacherDecision(
            feedback_type="corrective",
            confidence=0.99,
            interpretation="User correction.",
            training_target="주호입니다.",
            update_strength=0.9,
        ).model_copy(update=kwargs)
        self.received = []

    def decide(self, interaction):
        self.received.append(interaction)
        return self.decision


@pytest.fixture
def config(tmp_path):
    return tiny_config(tmp_path / "private")


@pytest.fixture
def student(config):
    return make_tiny_student(config)


@pytest.fixture
def teacher():
    return FixedTeacher()


@pytest.fixture
def engine(config, student, teacher):
    return TriadEngine(config, student, teacher)
