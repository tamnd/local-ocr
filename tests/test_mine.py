"""Mining training candidates out of adjudicated disagreements.

The interesting tests here are the exclusions. Producing a candidate from a
settled disagreement is bookkeeping; refusing to produce one from an illegible
verdict is the thing that keeps a fine tune honest, and it is the thing that
would be easy to lose in a refactor.
"""

from __future__ import annotations

import json

from local_ocr.mine import (
    SETTLED,
    Candidate,
    counts,
    from_record,
    mine,
    report,
    sidecars,
    to_jsonl,
)
from local_ocr.sidecar import Adjudication, Read, Record


def adjudication(**kw):
    base = dict(
        where="formula",
        what="span 3",
        first="\\aleph_0",
        second="\\aleph_1",
        severity="high",
        why="CDM 0.500",
        step="crop",
        winner="second",
        evidence="the strip reads aleph one",
        seconds=1.0,
        score=0.5,
    )
    base.update(kw)
    return Adjudication(**base)


def record(*items, page="0027"):
    got = Record(page=page, image_sha256="a" * 64, referee_ran=True, agreed=False)
    got.first = Read(reader="reader-a")
    got.second = Read(reader="reader-b")
    got.adjudicated.extend(items)
    return got


class TestFromRecord:
    def test_a_settled_disagreement_is_a_candidate(self):
        got = from_record(record(adjudication()), ("reader-a", "reader-b"))
        assert len(got) == 1
        assert got[0].page == "0027"
        assert got[0].image_sha256 == "a" * 64

    def test_the_loser_is_the_reader_to_train(self):
        # The referee won, so the primary is the one that got it wrong, and the
        # example belongs on the primary's curriculum and nobody else's.
        got = from_record(record(adjudication(winner="second")), ("reader-a", "reader-b"))
        assert got[0].for_reader == "reader-a"
        assert got[0].wrong == "\\aleph_0"
        assert got[0].right == "\\aleph_1"

    def test_the_other_way_round(self):
        got = from_record(record(adjudication(winner="first")), ("reader-a", "reader-b"))
        assert got[0].for_reader == "reader-b"
        assert got[0].wrong == "\\aleph_1"
        assert got[0].right == "\\aleph_0"

    def test_an_illegible_verdict_is_not_a_candidate(self):
        # The one exclusion that matters. Nobody knows what the page says there
        # and training on a guess is how a model learns to be confidently wrong.
        assert from_record(record(adjudication(step="illegible", winner="neither"))) == []

    def test_an_unspent_difference_is_not_a_candidate(self):
        assert from_record(record(adjudication(step="budget", winner=""))) == []

    def test_no_verdict_is_not_a_candidate(self):
        assert from_record(record(adjudication(winner=""))) == []

    def test_a_structural_difference_is_not_a_candidate(self):
        # The label would be a count and a reader cannot be trained toward one.
        assert from_record(record(adjudication(where="structure", what="heading count"))) == []

    def test_prose_is_a_candidate(self):
        got = from_record(record(adjudication(where="prose", what="prose sentence 4")))
        assert len(got) == 1

    def test_an_empty_side_is_not_a_candidate(self):
        assert from_record(record(adjudication(first="   "))) == []
        assert from_record(record(adjudication(second=""))) == []

    def test_the_rung_is_carried(self):
        got = from_record(record(adjudication(step="reread")))
        assert got[0].step == "reread"

    def test_both_settled_rungs_are_accepted(self):
        for step in SETTLED:
            assert from_record(record(adjudication(step=step)))

    def test_several_in_one_record(self):
        got = from_record(
            record(
                adjudication(what="span 1"),
                adjudication(what="span 2", winner="first"),
                adjudication(what="span 3", step="illegible", winner="neither"),
            ),
            ("reader-a", "reader-b"),
        )
        assert len(got) == 2
        assert {c.for_reader for c in got} == {"reader-a", "reader-b"}


class TestMine:
    def write(self, tmp_path, name, rec):
        rec.write(tmp_path / name)

    def test_it_finds_sidecars(self, tmp_path):
        self.write(tmp_path, "0027.md", record(adjudication()))
        self.write(tmp_path, "0028.md", record(adjudication(), page="0028"))
        assert len(list(sidecars(tmp_path))) == 2

    def test_it_looks_in_subdirectories(self, tmp_path):
        book = tmp_path / "ens-i-iv"
        book.mkdir()
        self.write(book, "0027.md", record(adjudication()))
        assert len(mine(tmp_path)) == 1

    def test_it_names_the_readers_from_the_sidecar(self, tmp_path):
        self.write(tmp_path, "0027.md", record(adjudication()))
        assert mine(tmp_path)[0].for_reader == "reader-a"

    def test_a_broken_sidecar_costs_one_page_and_not_the_run(self, tmp_path):
        self.write(tmp_path, "0027.md", record(adjudication()))
        (tmp_path / "0028.ocr.json").write_text("{not json")
        assert len(mine(tmp_path)) == 1

    def test_nothing_under_an_empty_directory(self, tmp_path):
        assert mine(tmp_path) == []

    def test_a_page_nobody_argued_about_yields_nothing(self, tmp_path):
        quiet = Record(page="0029", image_sha256="b" * 64, referee_ran=True, agreed=True)
        quiet.write(tmp_path / "0029.md")
        assert mine(tmp_path) == []


class TestCounts:
    def make(self, reader, where):
        return Candidate(
            page="0027",
            image_sha256="a" * 64,
            for_reader=reader,
            wrong="x",
            right="y",
            where=where,
            what="span 0",
            severity="high",
            step="crop",
            evidence="",
        )

    def test_split_by_reader_and_kind(self):
        got = counts(
            [
                self.make("reader-a", "formula"),
                self.make("reader-a", "formula"),
                self.make("reader-a", "prose"),
                self.make("reader-b", "formula"),
            ]
        )
        assert got["reader-a"] == {"formula": 2, "prose": 1, "total": 3}
        assert got["reader-b"] == {"formula": 1, "total": 1}

    def test_empty(self):
        assert counts([]) == {}


class TestOutput:
    def one(self):
        return Candidate(
            page="0027",
            image_sha256="a" * 64,
            for_reader="reader-a",
            wrong="\\aleph_0",
            right="\\aleph_1",
            where="formula",
            what="span 3",
            severity="high",
            step="crop",
            evidence="the strip reads aleph one",
            score=0.5,
        )

    def test_jsonl_is_one_object_a_line(self):
        text = to_jsonl([self.one(), self.one()])
        lines = [line for line in text.splitlines() if line]
        assert len(lines) == 2
        assert json.loads(lines[0])["for_reader"] == "reader-a"

    def test_jsonl_keeps_the_backslashes(self):
        got = json.loads(to_jsonl([self.one()]).splitlines()[0])
        assert got["wrong"] == "\\aleph_0"
        assert got["right"] == "\\aleph_1"

    def test_the_report_says_none_when_there_are_none(self):
        text = report([])
        assert "None." in text
        assert text.endswith("\n")

    def test_the_report_has_a_row_a_reader(self):
        text = report([self.one()])
        assert "| reader-a | 1 | 0 | 1 |" in text

    def test_the_report_counts_the_hard_ones(self):
        hard = Candidate(**{**self.one().__dict__, "step": "reread"})
        text = report([self.one(), hard])
        assert "1 of them took a second, tighter crop" in text
