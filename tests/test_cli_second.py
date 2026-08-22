"""Switching the referee on, and what a run leaves behind when it is on.

The wiring is the part of M6 most likely to be wrong in a way no unit test of
`second.py` would notice, because the fleet cannot pass a flag and so every
decision here is made by reading the environment. A variable read under the
wrong name is a referee that silently never runs, and the pages come out looking
exactly the same.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from local_ocr.batch import Options
from local_ocr.cli import (
    _budget,
    _head,
    _head_grammar,
    _head_label,
    _reader,
    _sidecars,
    main,
    mine_cmd,
    ocr_batch,
)
from local_ocr.second import SecondPass
from local_ocr.sidecar import SUFFIX, Record

PAGE = """18 ALGEBRAIC STRUCTURES Ch. I

Let $G$ be a group with identity $e$. Then $\\aleph_0$ is the cardinal of
$\\mathbf{N}$, and the centre of $G$ is a subgroup.
"""

OTHER = PAGE.replace("\\aleph_0", "\\aleph_1")


def args(**kw) -> argparse.Namespace:
    base = dict(
        backend="echo",
        model="reader-a",
        base_url="http://127.0.0.1:8801/v1",
    )
    base.update(kw)
    return argparse.Namespace(**base)


class Stub:
    """A reader that answers with one text and counts what it was asked."""

    def __init__(self, answer: str = PAGE) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def read(self, image: Path, prompt: str) -> str:
        self.asked.append((image.name, prompt))
        return self.answer


def lines() -> list[str]:
    return []


class TestBudget:
    def test_the_default_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCAL_OCR_BUDGET", raising=False)
        assert _budget(3) == 3

    def test_a_number_is_taken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_BUDGET", "7")
        assert _budget(3) == 7

    def test_zero_is_a_real_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Compare but never spend, which is how the catch rate is measured
        # without paying for a single crop.
        monkeypatch.setenv("LOCAL_OCR_BUDGET", "0")
        assert _budget(3) == 0

    def test_nonsense_falls_back_rather_than_failing_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OCR_BUDGET", "three")
        assert _budget(3) == 3

    def test_a_negative_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_BUDGET", "-1")
        assert _budget(3) == 3


class TestReader:
    def test_no_referee_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCAL_OCR_REFEREE", raising=False)
        assert not isinstance(_reader(args(), lines().append), SecondPass)

    def test_naming_one_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        got = _reader(args(), lines().append)
        assert isinstance(got, SecondPass)
        assert got.second is not None

    def test_the_sidecar_names_come_from_the_shortlist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The point of reading models.toml here rather than at write time: the
        # sidecar has to say which weights read the page, and the served name
        # alone does not.
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        got = _reader(args(), lines().append)
        assert got.names == ("reader-a", "reader-b")
        assert got.models[0].startswith("allenai/")
        assert got.models[1]
        assert got.revisions[0]

    def test_the_referee_gets_a_head_pass_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.headpass import HeadPass

        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        monkeypatch.delenv("LOCAL_OCR_HEAD_PASS", raising=False)
        got = _reader(args(), lines().append)
        assert isinstance(got.first, HeadPass)
        assert isinstance(got.second, HeadPass)

    def test_and_does_not_when_the_head_pass_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.headpass import HeadPass

        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        monkeypatch.setenv("LOCAL_OCR_HEAD_PASS", "0")
        got = _reader(args(), lines().append)
        assert not isinstance(got.first, HeadPass)
        assert not isinstance(got.second, HeadPass)

    def test_the_budget_is_carried_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        monkeypatch.setenv("LOCAL_OCR_BUDGET", "1")
        assert _reader(args(), lines().append).budget == 1

    def test_a_name_nobody_has_heard_of_costs_the_referee_and_not_the_pages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-z")
        monkeypatch.delenv("LOCAL_OCR_REFEREE_URL", raising=False)
        said: list[str] = []
        got = _reader(args(), said.append)
        assert not isinstance(got, SecondPass)
        assert any("reader-z" in line for line in said)

    def test_a_url_is_enough_without_a_shortlist_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # For a referee served by hand during a benchmark, which is most of them.
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "minerU")
        monkeypatch.setenv("LOCAL_OCR_REFEREE_URL", "http://127.0.0.1:8899/v1")
        got = _reader(args(), lines().append)
        assert isinstance(got, SecondPass)
        assert got.models[1] == ""


class TestCodexReferee:
    """The subscription as referee, which is the only one that needs no VRAM.

    Reader A takes 0.56 of the 4090 to hold its window, and the candidates that
    fit in what is left produce repeated tokens on a dense page. The ones that
    would not do that will not fit. Codex sidesteps the whole argument by not
    being on the card.
    """

    def test_codex_needs_no_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every other referee is a server and a port. This one is a subprocess,
        # so the URL check that guards the rest has to let it through.
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "codex")
        monkeypatch.delenv("LOCAL_OCR_REFEREE_URL", raising=False)
        got = _reader(args(), lines().append)
        assert isinstance(got, SecondPass)

    def test_the_default_model_is_the_full_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.backends.codex import MODEL

        monkeypatch.setenv("LOCAL_OCR_REFEREE", "codex")
        got = _reader(args(), lines().append)
        assert got.names[1] == MODEL

    def test_a_model_can_be_named_after_the_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # So a benchmark can pin an older revision without a code change.
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "codex:gpt-5.4-mini")
        got = _reader(args(), lines().append)
        assert got.names[1] == "gpt-5.4-mini"

    def test_it_is_a_codex_reader_underneath(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.backends.codex import CodexReader

        monkeypatch.setenv("LOCAL_OCR_REFEREE", "codex")
        monkeypatch.delenv("LOCAL_OCR_HEAD", raising=False)
        got = _reader(args(), lines().append)
        inner = got.second
        # A head pass may be wrapped round it; either way a CodexReader is at
        # the bottom, and no HTTP client is.
        assert isinstance(inner, CodexReader) or isinstance(
            getattr(inner, "inner", None), CodexReader
        )

    def test_a_missing_url_still_costs_the_referee_for_everyone_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The codex exception must not become a general one. A server referee
        # with no port is still a misconfiguration and should say so.
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-z")
        monkeypatch.delenv("LOCAL_OCR_REFEREE_URL", raising=False)
        said: list[str] = []
        assert not isinstance(_reader(args(), said.append), SecondPass)


class TestRun:
    """A whole batch with a referee behind it, on stub readers."""

    def run(self, tmp_path: Path, reader, capsys) -> str:
        src, dst = tmp_path / "in", tmp_path / "out"
        src.mkdir()
        Image.new("L", (1831, 2776), color=255).save(src / "0027.png")
        code = ocr_batch(
            [str(src), str(dst), "--ext", "png", "--prompt", "read this"],
            reader=reader,
        )
        assert code == 0
        return capsys.readouterr().out

    def test_it_writes_a_sidecar_beside_the_page(self, tmp_path: Path, capsys) -> None:
        self.run(tmp_path, SecondPass(Stub(), Stub()), capsys)
        assert (tmp_path / "out" / ("0027" + SUFFIX)).is_file()

    def test_the_sidecar_records_the_agreement(self, tmp_path: Path, capsys) -> None:
        self.run(tmp_path, SecondPass(Stub(), Stub()), capsys)
        raw = json.loads((tmp_path / "out" / ("0027" + SUFFIX)).read_text())
        assert raw["referee_ran"] is True
        assert raw["agreed"] is True
        assert raw["page"] == "0027"

    def test_a_disagreement_is_recorded(self, tmp_path: Path, capsys) -> None:
        self.run(tmp_path, SecondPass(Stub(PAGE), Stub(OTHER), budget=0), capsys)
        raw = json.loads((tmp_path / "out" / ("0027" + SUFFIX)).read_text())
        assert raw["agreed"] is False
        assert raw["unadjudicated"] >= 1

    def test_the_run_says_what_the_referee_cost(self, tmp_path: Path, capsys) -> None:
        out = self.run(tmp_path, SecondPass(Stub(), Stub()), capsys)
        assert "second reader: 1 pages, 0 disagreed" in out

    def test_no_referee_is_said_once(self, tmp_path: Path, capsys) -> None:
        out = self.run(tmp_path, SecondPass(Stub()), capsys)
        assert out.count("no referee configured") == 1

    def test_the_poller_cannot_see_the_sidecar(self, tmp_path: Path, capsys) -> None:
        # The Go side counts \.md$ and returns as soon as the count reaches the
        # page count. A sidecar that matched would end a batch early.
        self.run(tmp_path, SecondPass(Stub(), Stub()), capsys)
        names = sorted(p.name for p in (tmp_path / "out").iterdir())
        assert [n for n in names if n.endswith(".md")] == ["0027.md"]


class TestSidecars:
    def test_a_page_that_never_landed_gets_no_record(self, tmp_path: Path) -> None:
        # A refused page has no Markdown, and a record of how two readers failed
        # to produce one would be read by the miner as if it described a reading.
        src, dst = tmp_path / "in", tmp_path / "out"
        src.mkdir()
        dst.mkdir()
        image = src / "0027.png"
        image.write_bytes(b"")
        reader = SecondPass(Stub(), Stub())
        reader.records[image] = Record(page="0027")
        opts = Options(src=src, dst=dst)
        assert _sidecars(reader, opts, lines().append) == 0


class TestMineCommand:
    def sidecar(self, root: Path) -> None:
        from local_ocr.sidecar import Adjudication, Read, Record

        record = Record(page="0027", image_sha256="a" * 64, referee_ran=True, agreed=False)
        record.first = Read(reader="reader-a")
        record.second = Read(reader="reader-b")
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
            )
        )
        record.write(root / "0027.md")

    def test_it_prints_a_report(self, tmp_path: Path, capsys) -> None:
        self.sidecar(tmp_path)
        assert mine_cmd([str(tmp_path)]) == 0
        assert "| reader-a | 1 | 0 | 1 |" in capsys.readouterr().out

    def test_it_writes_jsonl(self, tmp_path: Path, capsys) -> None:
        self.sidecar(tmp_path)
        out = tmp_path / "pairs" / "candidates.jsonl"
        assert mine_cmd([str(tmp_path), "--jsonl", str(out)]) == 0
        assert json.loads(out.read_text().splitlines()[0])["for_reader"] == "reader-a"

    def test_it_writes_markdown(self, tmp_path: Path, capsys) -> None:
        self.sidecar(tmp_path)
        out = tmp_path / "report.md"
        assert mine_cmd([str(tmp_path), "--markdown", str(out)]) == 0
        assert "Training candidates" in out.read_text()

    def test_caught_reports_the_milestone_number(self, tmp_path: Path, capsys) -> None:
        self.sidecar(tmp_path)
        assert mine_cmd([str(tmp_path), "--caught"]) == 0
        out = capsys.readouterr().out
        assert "1 pages" in out
        assert "1 caught" in out

    def test_a_directory_that_is_not_one(self, tmp_path: Path, capsys) -> None:
        assert mine_cmd([str(tmp_path / "nowhere")]) == 2

    def test_it_is_reachable_from_main(self, tmp_path: Path, capsys) -> None:
        self.sidecar(tmp_path)
        assert main(["mine", str(tmp_path)]) == 0


class TestRefereePrompt:
    """Asking each reader in its own idiom, which is not optional in practice.

    Measured on gamingpc: MinerU2.5 given the eight kilobyte fleet prompt
    answered 114 of 124 pages with letter spaced text inside an array
    environment until it hit the token limit. The prompt is how a reader is
    asked; what is compared is what the two say about the page.
    """

    def test_nothing_set_means_both_are_asked_the_same(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-b")
        monkeypatch.delenv("LOCAL_OCR_REFEREE_PROMPT_FILE", raising=False)
        assert _reader(args(), lines().append).second_prompt == ""

    def test_a_file_is_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "reader-d.md"
        path.write_text("<image>\nFree OCR.")
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-d")
        monkeypatch.setenv("LOCAL_OCR_REFEREE_PROMPT_FILE", str(path))
        got = _reader(args(), lines().append)
        assert got.second_prompt == "<image>\nFree OCR."
        assert got.crop_prompt == "<image>\nFree OCR."

    def test_a_missing_file_costs_the_prompt_and_not_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OCR_REFEREE", "reader-d")
        monkeypatch.setenv("LOCAL_OCR_REFEREE_PROMPT_FILE", str(tmp_path / "nowhere.md"))
        said: list[str] = []
        got = _reader(args(), said.append)
        assert got.second_prompt == ""
        assert any("nowhere.md" in line for line in said)

    def test_the_referee_is_asked_its_own_way(self, tmp_path: Path) -> None:
        import asyncio

        first, second = Stub(PAGE), Stub(PAGE)
        pass_ = SecondPass(first, second, second_prompt="Free OCR.")
        image = tmp_path / "0027.png"
        Image.new("L", (1831, 2776), color=255).save(image)
        asyncio.run(pass_.read(image, "the long fleet prompt"))
        assert first.asked == [("0027.png", "the long fleet prompt")]
        assert second.asked == [("0027.png", "Free OCR.")]


class TestTheVolumeIsTold:
    """The grammar of a volume is known before the images are pushed.

    See the head pass module note: the guess learns a fact about a volume from
    fifteen contiguous pages of it. The Go side reads the grammar out of the
    corpus manifest and the batch is one volume, so it can just say.
    """

    def test_a_volume_that_prints_a_label_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_HEAD_LABEL", "1")
        assert _head_label() is True

    def test_a_volume_that_prints_none_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_HEAD_LABEL", "0")
        assert _head_label() is False

    def test_saying_nothing_leaves_the_guess_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCAL_OCR_HEAD_LABEL", raising=False)
        assert _head_label() is None

    def test_a_typo_leaves_the_guess_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """This arrives through a command line built by another program."""
        monkeypatch.setenv("LOCAL_OCR_HEAD_LABEL", "yes")
        assert _head_label() is None

    def test_it_reaches_the_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.headpass import HeadPass

        monkeypatch.setenv("LOCAL_OCR_HEAD_LABEL", "1")
        monkeypatch.delenv("LOCAL_OCR_HEAD_PASS", raising=False)
        monkeypatch.delenv("LOCAL_OCR_REFEREE", raising=False)
        got = _head(object())
        assert isinstance(got, HeadPass)
        assert got.head_label is True
        assert got.wants_label(), "told, so nothing has to be learned first"


class TestTheVolumeGrammarIsTold:
    """What the volume prints across the top of a page, in the manifest's word.

    A second question beside the label one and not the same question. The label
    is a yes or a no, and this is not: a numbered title across the top is the
    running head on a `head-number` volume and the body's section heading on a
    `foot-number` one, so a flag cannot carry it.
    """

    def test_the_manifest_word_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for word in ("head-number", "head-label", "foot-number"):
            monkeypatch.setenv("LOCAL_OCR_HEAD_GRAMMAR", word)
            assert _head_grammar() == word

    def test_saying_nothing_stays_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCAL_OCR_HEAD_GRAMMAR", raising=False)
        assert _head_grammar() == ""

    def test_whitespace_around_it_is_not_a_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_OCR_HEAD_GRAMMAR", "  foot-number\n")
        assert _head_grammar() == "foot-number"

    def test_a_typo_is_passed_through_and_refused_where_it_is_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The checking belongs next to the list, which is `headpass.GRAMMARS`."""
        from local_ocr.headpass import GRAMMARS, heading

        monkeypatch.setenv("LOCAL_OCR_HEAD_GRAMMAR", "footnumber")
        got = _head_grammar()
        assert got == "footnumber"
        assert got not in GRAMMARS
        assert not heading("5. IMAGE OF A SUMMABLE FAMILY", got)

    def test_it_reaches_the_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from local_ocr.headpass import HeadPass

        monkeypatch.setenv("LOCAL_OCR_HEAD_GRAMMAR", "foot-number")
        monkeypatch.delenv("LOCAL_OCR_HEAD_LABEL", raising=False)
        monkeypatch.delenv("LOCAL_OCR_HEAD_PASS", raising=False)
        monkeypatch.delenv("LOCAL_OCR_REFEREE", raising=False)
        got = _head(object())
        assert isinstance(got, HeadPass)
        assert got.grammar == "foot-number"

    def test_an_unclassified_volume_reaches_the_wrapper_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from local_ocr.headpass import HeadPass

        monkeypatch.delenv("LOCAL_OCR_HEAD_GRAMMAR", raising=False)
        monkeypatch.delenv("LOCAL_OCR_HEAD_PASS", raising=False)
        monkeypatch.delenv("LOCAL_OCR_REFEREE", raising=False)
        got = _head(object())
        assert isinstance(got, HeadPass)
        assert got.grammar == ""
