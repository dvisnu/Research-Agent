from pathlib import Path

import pytest

from research_agent.config import Settings
from research_agent.storage import Database
from research_agent.trace import TraceRecorder
from fakes import fake_services


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "var")


@pytest.fixture
def db(settings: Settings):
    with Database(settings.db_path) as db:
        yield db


@pytest.fixture
def recorder(db: Database) -> TraceRecorder:
    db.create_run("run_test", "a question")
    return TraceRecorder(db)


@pytest.fixture
def services(settings: Settings, tmp_path: Path):
    """Fake literature sources that find nothing, and a fake embedder."""
    services = fake_services(settings, tmp_path)
    yield services
    services.sources.close()
