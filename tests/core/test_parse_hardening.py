"""Parsing and decoding failures should surface as our errors, not escape raw.

Three narrow cases, each a place where a malformed input crossed a boundary the
surrounding code did not expect: a bracketed-but-invalid JSON payload, a
non-UTF8 session file, and a JSON payload of the wrong shape.
"""

import json

import pytest


class TestDatasetJsonPayloads:
    """The extractors raise ValueError for unparseable LLM output.

    Two sites pulled a JSON substring out of prose and then parsed it
    unguarded, so a payload with brackets but invalid contents escaped as a raw
    JSONDecodeError past the ValueError the surrounding code raises.
    """

    def test_bracketed_but_invalid_list_raises_our_error(self):
        from evolution.core.dataset_builder import _extract_json_list

        with pytest.raises(ValueError, match="Could not parse"):
            _extract_json_list("here you go: [{'bad': quotes,}] hope that helps")

    def test_bracketed_but_invalid_object_raises_our_error(self):
        from evolution.core.dataset_builder import _extract_json_object

        with pytest.raises(ValueError, match="Could not parse"):
            _extract_json_object("sure: {trailing: comma,} done")

    def test_valid_json_of_the_wrong_shape_raises_our_error(self):
        """Decoding is not the only way a payload can be unusable.

        A bare list of ints parses fine and then fails much later with an
        AttributeError when something calls .get on an int -- far from the cause.
        """
        from evolution.core.dataset_builder import _extract_json_list

        with pytest.raises(ValueError, match="shape"):
            _extract_json_list("[1, 2, 3]")

    def test_well_formed_payloads_still_parse(self):
        from evolution.core.dataset_builder import _extract_json_list, _extract_json_object

        assert _extract_json_list('prose [{"a": 1}] more') == [{"a": 1}]
        assert _extract_json_object('prose {"a": 1} more') == {"a": 1}


class TestNonUtf8SessionFiles:
    """A stray non-UTF8 file must be skipped, not crash the importer.

    UnicodeDecodeError derives from ValueError, not OSError, so it escaped the
    guards at both read sites -- and the Claude Code history log is the likelier
    of the two to be long-lived and mixed-encoding.
    """

    def test_legacy_session_file_is_skipped(self, tmp_path, monkeypatch):
        from evolution.core.external_importers import HermesSessionImporter, iter_hermes_sessions

        (tmp_path / "bad.json").write_bytes(b'{"messages": [{"x": "\xff\xfe"}]}')
        (tmp_path / "good.json").write_text(
            json.dumps({"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
        )
        monkeypatch.setattr(HermesSessionImporter, "SESSION_DIR", tmp_path)
        monkeypatch.setattr(HermesSessionImporter, "STATE_DB", tmp_path / "missing.db")

        sessions = dict(iter_hermes_sessions())

        assert "s1" in sessions, "a valid sibling must still be yielded"

    def test_history_log_line_is_skipped(self, tmp_path, monkeypatch):
        """Decoding happens during line iteration, outside the json guard."""
        from evolution.core.external_importers import ClaudeCodeImporter

        path = tmp_path / "history.jsonl"
        # Long enough to survive the importer's own minimum-length filter, so the
        # test is about decoding rather than about that filter.
        first, second = "summarise this transcript", "categorise these messages"
        path.write_bytes(
            json.dumps({"display": first}).encode()
            + b"\n\xff\xfe bad bytes\n"
            + json.dumps({"display": second}).encode()
            + b"\n"
        )
        monkeypatch.setattr(ClaudeCodeImporter, "HISTORY_PATH", path)

        messages = ClaudeCodeImporter.extract_messages()

        texts = [m.get("task_input") for m in messages]
        assert first in texts and second in texts


class TestMiproFallbackValset:
    """The rescue path should not lose the held-out split — or crash on an empty one."""

    def test_valset_reaches_the_optimizer(self, monkeypatch):
        """GEPA passes its valset; the fallback silently did not.

        Ours is a genuine named split, not a slice of trainset, so passing it
        replaces the optimizer's internal trainset-derived split rather than
        feeding training data back in.
        """
        import dspy

        from evolution.skills.evolve_skill import _default_mipro_runner

        seen = {}

        class _FakeMIPRO:
            def __init__(self, **kwargs):
                pass

            def compile(self, module, **kwargs):
                seen.update(kwargs)
                return module

        monkeypatch.setattr(dspy, "MIPROv2", _FakeMIPRO)
        _default_mipro_runner(
            baseline_module=object(), trainset=[1, 2], metric=lambda *a, **k: 1.0,
            seed=42, valset=[3, 4],
        )

        assert seen.get("valset") == [3, 4]

    def test_empty_valset_takes_the_internal_split(self, monkeypatch):
        """The optimizer rejects a non-None empty valset.

        Passing one straight through would turn the path that exists to survive a
        GEPA failure into a hard crash, so an empty valset must become None and
        keep today's behavior.
        """
        import dspy

        from evolution.skills.evolve_skill import _default_mipro_runner

        seen = {}

        class _FakeMIPRO:
            def __init__(self, **kwargs):
                pass

            def compile(self, module, **kwargs):
                seen.update(kwargs)
                return module

        monkeypatch.setattr(dspy, "MIPROv2", _FakeMIPRO)
        _default_mipro_runner(
            baseline_module=object(), trainset=[1, 2], metric=lambda *a, **k: 1.0,
            seed=42, valset=[],
        )

        assert seen.get("valset") is None
