"""Shared fixtures for the tool-evolution tests."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes_state_db(tmp_path, monkeypatch):
    """``iter_hermes_sessions`` reads ``~/.hermes/state.db`` first; a real db on the
    dev/CI box would hijack the JSON-fixture and dry-run tests. Default ``STATE_DB``
    to an absent path for every tools test; tests that need a populated db override
    it explicitly (``patch.object(HermesSessionImporter, "STATE_DB", ...)``)."""
    from evolution.core.external_importers import HermesSessionImporter

    monkeypatch.setattr(HermesSessionImporter, "STATE_DB", tmp_path / "no_state.db")
