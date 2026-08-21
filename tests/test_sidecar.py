"""The record beside every page.

Two things are worth testing here and they are not the obvious one. Writing a
JSON file is not interesting. What is interesting is that the sidecar survives a
round trip through disk without losing the fields the miner depends on, and that
`_clean` drops the noise without dropping a false, because `agreed: false` and
`referee_ran: false` are the two most load bearing values a sidecar carries and
an eager cleaner would silently delete both.
"""

from __future__ import annotations

import json

from local_ocr.compare import Difference, Severity, Where
from local_ocr.sidecar import (
    SUFFIX,
    VERSION,
    Adjudication,
    Read,
    Record,
    as_adjudication,
    digest,
    load,
    sidecar_for,
    text_digest,
)


class TestNaming:
    def test_a_page_gets_a_sidecar_beside_it(self, tmp_path):
        got = sidecar_for(tmp_path / "0027.md")
        assert got.name == "0027" + SUFFIX
        assert got.parent == tmp_path

    def test_a_name_without_a_suffix_still_gets_one(self, tmp_path):
        assert sidecar_for(tmp_path / "0027").name == "0027" + SUFFIX

    def test_it_sorts_next_to_its_page(self):
        # 0027.md and 0027.ocr.json sort adjacent, and 0028.md comes after both.
        names = sorted(["0027.md", "0027.ocr.json", "0028.md", "0026.md"])
        assert names == ["0026.md", "0027.md", "0027.ocr.json", "0028.md"]


class TestDigest:
    def test_the_same_bytes_give_the_same_hash(self, tmp_path):
        one, other = tmp_path / "a", tmp_path / "b"
        one.write_bytes(b"the same page")
        other.write_bytes(b"the same page")
        assert digest(one) == digest(other)

    def test_different_bytes_do_not(self, tmp_path):
        one, other = tmp_path / "a", tmp_path / "b"
        one.write_bytes(b"300 dpi")
        other.write_bytes(b"600 dpi")
        assert digest(one) != digest(other)

    def test_a_large_file_is_read_in_blocks(self, tmp_path):
        big = tmp_path / "big"
        big.write_bytes(b"x" * (3 << 20))
        assert len(digest(big)) == 64

    def test_text_digest_is_stable(self):
        assert text_digest("read this page") == text_digest("read this page")


class TestClean:
    def test_empty_strings_go(self, tmp_path):
        record = Record(page="0027", image_sha256="abc")
        raw = json.loads(record.to_json())
        assert "gates" not in raw
        assert raw["page"] == "0027"

    def test_false_stays(self):
        record = Record(page="0027", referee_ran=False, agreed=False)
        raw = json.loads(record.to_json())
        assert raw["referee_ran"] is False
        assert raw["agreed"] is False

    def test_true_stays(self):
        raw = json.loads(Record(referee_ran=True, agreed=True).to_json())
        assert raw["referee_ran"] is True
        assert raw["agreed"] is True

    def test_a_zero_count_stays_inside_counts(self):
        record = Record(counts={"high": 0, "medium": 2, "low": 0})
        raw = json.loads(record.to_json())
        assert raw["counts"] == {"high": 0, "medium": 2, "low": 0}


class TestRoundTrip:
    def make(self):
        record = Record(
            page="0027",
            image_sha256="a" * 64,
            width=1831,
            height=2776,
            dpi=300,
            referee_ran=True,
            agreed=False,
            counts={"high": 1, "medium": 0, "low": 3},
            unadjudicated=2,
            chose="second",
            gates={"head": "ok", "math": "unclosed span on line 4"},
        )
        record.first = Read(
            reader="reader-a",
            model="allenai/olmOCR-2-7B-1025-FP8",
            revision="40bd7202494b8264ee17ada08b401b5aab7a9ce1",
            seconds=8.4,
            text_sha256="b" * 64,
        )
        record.second = Read(reader="reader-b", model="opendatalab/MinerU2.5-2509-1.2B")
        record.adjudicated.append(
            Adjudication(
                where="formula",
                what="span 3",
                first="\\aleph_0",
                second="\\aleph_1",
                severity="high",
                why="CDM 0.500",
                step="crop",
                winner="second",
                evidence="the strip reads aleph one",
                seconds=2.1,
                score=0.5,
            )
        )
        return record

    def test_survives_disk(self, tmp_path):
        answer = tmp_path / "0027.md"
        path = self.make().write(answer)
        back = load(path)
        assert back.page == "0027"
        assert back.version == VERSION
        assert back.referee_ran is True
        assert back.agreed is False
        assert back.chose == "second"
        assert back.dpi == 300
        assert back.unadjudicated == 2

    def test_the_readers_survive(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        back = load(path)
        assert back.first is not None
        assert back.first.reader == "reader-a"
        assert back.first.revision.startswith("40bd7202")
        assert back.second is not None
        assert back.second.reader == "reader-b"

    def test_the_adjudication_survives(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        back = load(path)
        assert len(back.adjudicated) == 1
        one = back.adjudicated[0]
        assert one.winner == "second"
        assert one.step == "crop"
        assert one.score == 0.5

    def test_the_gates_survive(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        assert load(path).gates["head"] == "ok"

    def test_an_unknown_field_is_ignored_rather_than_fatal(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        raw = json.loads(path.read_text())
        raw["something_from_the_future"] = {"nested": True}
        raw["first"]["also_new"] = 3
        path.write_text(json.dumps(raw))
        back = load(path)
        assert back.page == "0027"
        assert back.first is not None
        assert back.first.reader == "reader-a"

    def test_a_wrong_type_does_not_overwrite_a_default(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        raw = json.loads(path.read_text())
        raw["dpi"] = "three hundred"
        path.write_text(json.dumps(raw))
        assert load(path).dpi == 0

    def test_the_file_ends_in_a_newline(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        assert path.read_text().endswith("\n")

    def test_it_is_readable_by_a_person(self, tmp_path):
        path = self.make().write(tmp_path / "0027.md")
        text = path.read_text()
        assert "\n  " in text  # indented
        assert '"page"' in text


class TestAsAdjudication:
    def test_it_carries_the_difference(self):
        one = Difference(
            Where.FORMULA, "span 3", "\\aleph_0", "\\aleph_1", Severity.HIGH, "CDM 0.500", 0.5
        )
        got = as_adjudication(one, step="crop", winner="second", evidence="aleph one", seconds=1.5)
        assert got.where == "formula"
        assert got.what == "span 3"
        assert got.severity == "high"
        assert got.score == 0.5
        assert got.winner == "second"
        assert got.evidence == "aleph one"
