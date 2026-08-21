"""Reading a page through the local ChatGPT subscription.

The reader itself is four lines of subprocess; the part worth testing is the
transcript parsing, because codex prints the reply twice and the two copies are
not always the same. Taking the wrong one gives a page that is subtly truncated
and passes every acceptance rule, which is the exact failure class M6 exists to
catch.

No subprocess is spawned here. `codex` is not on a CI box and a test that needs
a live subscription is a test that gets skipped and then deleted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_ocr.backends.codex import MODEL, CodexReader, answer
from local_ocr.batch import Refused

# A transcript with the shape codex actually produces. Banner, the word user,
# the prompt echoed, the word codex, the reply streamed, the marker, the count,
# and the reply again. Both copies are here and they differ, which is the whole
# reason `answer` exists: the streamed copy has lost the closing dollar.
TRANSCRIPT = """\
OpenAI Codex v0.55.0
--------
workdir: /tmp
model: gpt-5.4
--------
user
Transcribe this page.

codex
## 3. DISTRIBUTIVITY

Let $(X_i)_{i \\in I$
tokens used
13,966
## 3. DISTRIBUTIVITY

Let $(X_i)_{i \\in I}$ be a family of sets.
"""


def read(reader: CodexReader, image: Path, prompt: str = "read it") -> str:
    return asyncio.run(reader.read(image, prompt))


def stub(monkeypatch, *, stdout: str, returncode: int = 0):
    """Replace the subprocess with a canned transcript."""

    class Proc:
        def __init__(self):
            self.returncode = returncode

        async def communicate(self):
            return stdout.encode(), b""

        def kill(self):
            pass

        async def wait(self):
            return returncode

    seen: dict[str, tuple] = {}

    async def fake(*argv, **kwargs):
        seen["argv"] = argv
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return seen


@pytest.fixture
def page(tmp_path: Path) -> Path:
    p = tmp_path / "0113.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


# ---------------------------------------------------------------------------
# The transcript


def test_the_reply_after_the_count_is_the_one_codex_meant():
    text, tokens = answer(TRANSCRIPT)
    assert tokens == 13966
    assert text.endswith("be a family of sets.")
    # The streamed copy above is truncated mid-formula. Taking it would produce
    # a page with an unbalanced dollar, and worse, a page missing a clause.
    assert "\\in I}$ be a family" in text


def test_the_prompt_echo_is_not_part_of_the_answer():
    text, _ = answer(TRANSCRIPT)
    assert "Transcribe this page." not in text
    assert "workdir" not in text


def test_a_transcript_with_no_marker_comes_back_whole():
    """Codex died early. Hand the text on and let the rules reject it.

    Better a page rejected with a strange body somebody can read than a page
    silently dropped with no record of why.
    """
    text, tokens = answer("error: something went wrong\n")
    assert tokens == 0
    assert "something went wrong" in text


def test_the_last_marker_wins():
    """The word appears in the page when the page is about tokens."""
    doubled = "codex\nthe tokens used here\ntokens used\n42\nthe real answer\n"
    text, tokens = answer(doubled)
    assert tokens == 42
    assert text == "the real answer"


def test_a_count_with_thousands_separators_is_a_number():
    _, tokens = answer("tokens used\n1,234,567\nbody\n")
    assert tokens == 1234567


def test_an_unreadable_count_is_zero_rather_than_a_crash():
    text, tokens = answer("tokens used\nlots\nbody\n")
    assert tokens == 0
    assert text == "body"


# ---------------------------------------------------------------------------
# The command line


def test_the_separator_is_there(monkeypatch, page):
    """Without it, `-i` swallows the prompt as a second image path.

    `-i` is variadic. `codex exec -i img "$prompt"` dies with "No prompt
    provided via stdin", which is a confusing way to say "that was two images
    and no question".
    """
    seen = stub(monkeypatch, stdout=TRANSCRIPT)
    read(CodexReader(), page, "read it")
    argv = list(seen["argv"])
    assert argv[argv.index("-i") + 1] == str(page)
    assert argv[argv.index("-i") + 2] == "--"
    assert argv[-1] == "read it"


def test_the_full_model_is_the_default(monkeypatch, page):
    """gpt-5.4, measured cheaper and faster and better than the mini at once.

    On a dense French page the mini burned 41721 tokens and returned eleven
    mathematical spans against the sixty four the reference has. The full model
    read the same page in 48 seconds for 7697 tokens and found fifty nine.
    """
    seen = stub(monkeypatch, stdout=TRANSCRIPT)
    read(CodexReader(), page)
    argv = list(seen["argv"])
    assert argv[argv.index("--model") + 1] == MODEL == "gpt-5.4"


def test_the_image_is_attached_rather_than_looked_for(monkeypatch, page):
    """Letting codex open the file costs three times as much for one reason.

    It becomes a shell tool call, and every tool turn re-sends the whole
    transcript. 42033 tokens against 14102, and 309 seconds against 70.
    """
    seen = stub(monkeypatch, stdout=TRANSCRIPT)
    read(CodexReader(), page, "read it")
    argv = list(seen["argv"])
    assert "-i" in argv
    assert str(page) not in argv[-1]


# ---------------------------------------------------------------------------
# Refusing, which is a fact about the page, and failing, which is not


def test_a_missing_image_is_refused_before_anything_is_spawned(tmp_path):
    with pytest.raises(Refused, match="no image"):
        read(CodexReader(), tmp_path / "nope.png")


def test_a_nonzero_exit_is_a_refusal(monkeypatch, page):
    stub(monkeypatch, stdout="codex: not logged in\n", returncode=1)
    with pytest.raises(Refused, match="not logged in"):
        read(CodexReader(), page)


def test_the_variadic_mistake_is_recognised_even_on_a_zero_exit(monkeypatch, page):
    stub(monkeypatch, stdout="tokens used\n0\nNo prompt provided via stdin.\n")
    with pytest.raises(Refused, match="failed"):
        read(CodexReader(), page)


def test_a_model_the_subscription_does_not_carry_is_a_refusal(monkeypatch, page):
    stub(
        monkeypatch,
        stdout="tokens used\n0\nnot supported when using Codex with a ChatGPT account\n",
    )
    with pytest.raises(Refused, match="failed"):
        read(CodexReader(), page)


def test_a_content_refusal_says_so(monkeypatch, page):
    stub(monkeypatch, stdout="tokens used\n900\nI'm sorry, I can't help with that.\n")
    with pytest.raises(Refused, match="declined"):
        read(CodexReader(), page)


def test_an_empty_answer_is_a_refusal(monkeypatch, page):
    stub(monkeypatch, stdout="tokens used\n900\n\n")
    with pytest.raises(Refused, match="nothing"):
        read(CodexReader(), page)


# ---------------------------------------------------------------------------
# What it cost, which is the number this reader has to justify itself with


def test_the_tokens_are_counted_across_pages(monkeypatch, page):
    reader = CodexReader()
    stub(monkeypatch, stdout=TRANSCRIPT)
    read(reader, page)
    read(reader, page)
    assert reader.pages == 2
    assert reader.tokens == 2 * 13966
    assert "27932 tokens" in reader.cost()
    assert "13966 a page" in reader.cost()


def test_a_refused_page_still_counts_what_it_cost(monkeypatch, page):
    """A page that failed after the model read it was not free."""
    reader = CodexReader()
    stub(monkeypatch, stdout="tokens used\n5000\nI cannot assist with this.\n")
    with pytest.raises(Refused):
        read(reader, page)
    assert reader.tokens == 5000


def test_a_reader_that_never_ran_says_nothing(monkeypatch):
    assert CodexReader().cost() == ""
