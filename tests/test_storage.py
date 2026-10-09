import sqlite3

import pytest

from research_agent.storage import Database, parse_sentence_id, section_id, sentence_id, utc_now
from research_agent.storage.db import SCHEMA_VERSION


def test_sentence_address_round_trip():
    address = sentence_id("p_abc", 3, 12)
    assert address == "p_abc:s3:12"
    assert address.startswith(section_id("p_abc", 3) + ":")
    assert parse_sentence_id(address) == ("p_abc", 3, 12)


@pytest.mark.parametrize("bad", ["p_abc", "p_abc:s3", "p_abc:3:12", "p_abc:sx:1"])
def test_parse_sentence_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_sentence_id(bad)


def test_schema_is_created_once(settings):
    with Database(settings.db_path) as db:
        db.create_run("run_1", "q")
    with Database(settings.db_path) as db:  # reopening must not recreate tables
        assert db.query("PRAGMA user_version")[0][0] == SCHEMA_VERSION
        assert db.get_run("run_1")["status"] == "running"


def test_unknown_schema_version_is_refused(settings):
    with Database(settings.db_path) as db:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="schema version"):
        Database(settings.db_path)


def test_sentences_need_an_existing_section(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO sentences (sentence_id, section_id, idx, text, char_start, char_end)"
            " VALUES ('p_x:s0:0', 'p_x:s0', 0, 'text', 0, 4)",
        )


def test_paper_section_sentence_chain(db):
    db.execute(
        "INSERT INTO papers (paper_id, title, arxiv_id, created_at) VALUES (?, ?, ?, ?)",
        ("p_1", "A paper", "2106.09685", utc_now()),
    )
    text = "First sentence. Second sentence."
    db.execute(
        "INSERT INTO sections (section_id, paper_id, idx, heading, text) VALUES (?, ?, 0, ?, ?)",
        (section_id("p_1", 0), "p_1", "Introduction", text),
    )
    db.execute(
        "INSERT INTO sentences (section_id, sentence_id, idx, text, char_start, char_end)"
        " VALUES (?, ?, 1, ?, 16, 32)",
        (section_id("p_1", 0), sentence_id("p_1", 0, 1), "Second sentence."),
    )
    row = db.query("SELECT text, char_start, char_end FROM sentences")[0]
    assert text[row["char_start"] : row["char_end"]] == row["text"]


def test_trace_events_are_append_only(db, recorder):
    with recorder.step("run_test", "brief"):
        pass
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE trace_events SET status = 'error'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("DELETE FROM trace_events")
