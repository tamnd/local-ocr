"""The ways a model's answer is not what was asked for.

A transcription of the `textguard` package in `tamnd/bourbaki-solver`. A model
handed a page image sometimes talks instead of transcribing: it apologises, it
announces what it is about to do, it summarises, or it hands the prompt back.
All of that is fluent English, none of it is Bourbaki, and once it is written to
a page file it looks exactly like text that was read off the page.

The phrase lists are copied verbatim and in order. Order is load bearing in two
ways: `check` reports one leak per line and takes the first kind that matches,
so a line that both apologises and narrates is a refusal and is counted once,
and within a kind the first matching phrase is the one reported. A list sorted
for tidiness would change what the failures report says.

Every list here is narrow on purpose, and each narrowing has a page behind it.
The bare word "violates" rejected a real page of chapter IV, because a theorem
says a map violates no relation. "we do not see" is ordinary mathematical prose.
"in summary" is left out entirely because Bourbaki writes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_ocr.rules.goquote import quote
from local_ocr.rules.mathtex import in_math


@dataclass(frozen=True)
class Leak:
    """One thing found in an answer that should not be there."""

    kind: str
    """refusal, no-image, meta, prompt, thinking, markup or empty."""
    detail: str
    line: int
    """One based, and 0 when the whole answer is at fault."""


# The model declining. A page that opens with one of these has no transcription
# in it at all.
REFUSALS = (
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i apologise",
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "unable to assist",
    "can't help with",
    "cannot help with",
    "as an ai",
    "as a language model",
    "i'm just an ai",
    "against my guidelines",
    # Narrow on purpose. A refusal says the request violates something; a
    # theorem says a map violates no relation, and the bare word rejected a real
    # page of chapter IV the first time this ran.
    "violates our",
    "violates my",
    "violates the content",
)

# The model answering politely to a message that arrived without its attachment.
# Not a refusal and not narration: the prompt got through, the page did not, and
# the answer is a browser fix rather than another call at 151 seconds. It cost
# three pages of the first live run, and the first person is what keeps it from
# firing on "the image is not attached to any choice of basis".
NO_IMAGE = (
    "i don't see an image",
    "i don't see the image",
    "i don't see any image",
    "i do not see an image",
    "i do not see the image",
    "i didn't receive an image",
    "i did not receive an image",
    "no image was attached",
    "there is no image attached",
    "please upload the",
    "please upload an image",
    "please attach the image",
    "upload the page image",
    "you want transcribed",
)

# The model narrating. Worse than a refusal, because the transcription usually
# follows and the page looks almost right.
METAS = (
    "here is the transcription",
    "here's the transcription",
    "here is the text",
    "here's the text",
    "here is the transcribed",
    "the image shows",
    "the image contains",
    "this image appears",
    "this page appears to be",
    "sure, here",
    "sure! here",
    "certainly, here",
    "certainly! here",
    "below is the transcription",
    "i have transcribed",
    "transcription of the image",
    "let me know if",
    "hope this helps",
    "note that i have",
)

# The model handing back its instructions rather than its answer. The second
# group is the solve side, and it cost six files: two solutions of the Theory of
# Sets went into the corpus containing the repair prompt read back, its
# headings, its rules and its placeholder tag line, and nothing else.
PROMPTS = (
    "transcribe the complete text",
    "render all mathematical expressions as latex",
    "output only the raw transcribed content",
    "do not summarize, paraphrase",
    "the solution as it stands",
    "what the judges said",
    "do not write a list of changes",
    "a note about what you fixed",
    "uses: xxxx",
    "xxxx and yyyy",
)

# The model handing back its reasoning. There is no answer under it at all,
# which is what separates it from a meta line with a transcription following.
# The opener is the whole of what is looked for, because every line after the
# first is ordinary English about the problem and so is a solution.
THINKING = (
    "here's a thinking process",
    "here is a thinking process",
    "here's my thinking process",
    "here is my thinking process",
    "let me think through this step by step",
)

# The provider's own formatting, wrapped around an answer that is otherwise
# fine, and the nastier failure because there is no English sentence to search
# for. A retranslation of the appendix on the Nullstellensatz reached the corpus
# inside a directive fence and passed all seven translation rules; it was found
# by reading the diff. None of the three can be part of a page of Bourbaki, so
# all three are refused wherever they turn up and not only at the start.
MARKUP = (
    (
        "a directive fence, which is the provider's markup and not Markdown this corpus uses",
        re.compile(r"^\s*:::", re.M | re.A),
    ),
    ("a citation anchor", re.compile(r"【[^】]*】|oai_citation|contentReference")),
    (
        "a private use character, which is a provider's own marker",
        re.compile(r"[\ue000-\uf8ff]"),
    ),
)

_KINDS = (
    ("refusal", REFUSALS),
    ("no-image", NO_IMAGE),
    ("prompt", PROMPTS),
    ("thinking", THINKING),
    ("meta", METAS),
)

# A model writes I don't and I'm sorry with U+2019, and every phrase above is
# spelled with U+0027, so none of them matched. That is why "I don't see an
# image attached" reached the corpus with no leak reported at all. The answer
# itself is untouched, because the apostrophe a page prints is the page's own
# business.
_STRAIGHTEN = str.maketrans({"\u2019": "'", "\u2018": "'", "\uff07": "'"})


def _straighten(text: str) -> str:
    return text.translate(_STRAIGHTEN)


def check(text: str) -> list[Leak]:
    """Read an answer and report everything wrong with it.

    Every leak is reported rather than the first, because a page that both
    apologises and narrates is a different failure from one that only narrates,
    and the retry that follows is chosen from what was found.
    """
    if not text.strip():
        return [Leak("empty", "the answer is empty", 0)]
    leaks: list[Leak] = []
    for i, line in enumerate(text.split("\n")):
        for what, pattern in MARKUP:
            found = pattern.search(line)
            if found is not None and found.group(0) != "":
                leaks.append(Leak("markup", what + ", " + quote(found.group(0).strip()), i + 1))
                break
    for i, line in enumerate(text.split("\n")):
        lower = _straighten(line.lower())
        for kind, phrases in _KINDS:
            hit = next((p for p in phrases if p in lower), None)
            if hit is not None:
                leaks.append(Leak(kind, hit, i + 1))
                break
    return leaks


def clean(text: str) -> bool:
    """Whether an answer is free of leaks."""
    return not check(text)


# The code fence a model wraps an answer in when it was asked for Markdown and
# decided to be helpful.
_FENCES = re.compile(r"\A```[a-zA-Z]*\n(.*)\n```\s*\Z", re.S | re.A)


def strip(text: str) -> str:
    """Remove the wrapping a model adds around an otherwise good answer.

    Not the same as `check`. A fenced answer is correct text in the wrong
    packaging and unwrapping it is safe; a narrated answer is text that may have
    been altered, and no amount of trimming makes that safe, so it is rejected
    rather than repaired.
    """
    trimmed = text.strip()
    match = _FENCES.match(trimmed)
    if match is not None:
        trimmed = match.group(1).strip()
    return trimmed


DISPLAY = "$$"
SPAN = "$"
STAR = "\\*"

_BRACKETS = (
    ("\\[", "a display opened with a bracket"),
    ("\\]", "a display closed with a bracket"),
    ("\\(", "a span opened with a parenthesis"),
    ("\\)", "a span closed with a parenthesis"),
)

_DELIMITERS = (("\\[", DISPLAY), ("\\]", DISPLAY), ("\\(", SPAN), ("\\)", SPAN))


def dollars(text: str) -> tuple[str, int]:
    """Write the corpus's delimiters, and say how many were turned round.

    The corpus writes a display between two pairs of dollars and a span between
    one pair, everywhere. LaTeX's other spelling means the same to a
    mathematician and nothing at all to this corpus, because every rule that
    reads mathematics here finds spans by their dollars.

    LOAD BEARING. The row break of a matrix is `\\\\`, and `\\\\[2pt]` is a row
    break asking for space after it. Both start with a backslash that is not a
    delimiter's, so they are put aside before the substitution and put back
    afterwards. The corpus has a good many of them and every one is a legitimate
    row break rather than a display.

    Idempotent, since a text this has been through carries no `\\[` for it to
    find on a second pass.
    """
    held = text.replace("\\\\", "\x00")
    n = sum(held.count(delim) for delim, _ in _BRACKETS)
    if n == 0:
        return text, 0
    for delim, into in _DELIMITERS:
        held = held.replace(delim, into)
    return held.replace("\x00", "\\\\"), n


# The glyphs a model hands back in the star's place, and there are four of them
# in this corpus. None of the OCR prompts said how to write the star, so the
# model chose by what the glyph looked like rather than by what it meant. Theory
# of Sets alone has 24 of them against 82 written properly, and one paragraph of
# chapter IV has both forms in it.
ORNAMENTS = {
    "\u2217": "an asterisk operator",
    "\u273b": "a teardrop spoked asterisk",
    "\u2733": "an eight spoked asterisk",
    "\u204e": "a low asterisk",
}


def _bare_star(rs: list[str], i: int, math: list[bool]) -> bool:
    """A bare ASCII asterisk where the corpus writes the escaped one.

    The test is the space on both sides, and it comes from the Elements. To the
    Reader says the passages are always placed between two asterisks, with the
    space on the inside. Markdown reads the same shape the other way round: an
    emphasis run has to open on a non space and close on a non space, so
    `*signs*` is emphasis and `* signs *` is not.

    A backslash, a second asterisk and a letter all fail this, so an escaped
    star, a bold run and an emphasis run are all left alone without being named
    separately.
    """
    if rs[i] != "*" or math[i]:
        return False

    def space(j: int) -> bool:
        if j < 0 or j >= len(rs):
            return True  # a line end counts, and so does the start of the body
        return rs[j] in (" ", "\n", "\t")

    return space(i - 1) and space(i + 1)


def stars(text: str) -> tuple[str, int]:
    """Put the corpus's star back wherever a model wrote something else.

    Outside the math spans only, and that is not tidiness. U+2217 inside the
    mathematics is a binary operation, a dual or a pullback. The ASCII asterisk
    is worse: inside a span it is a convolution, an adjoint, a dual basis or the
    units of a ring, and `K^*` runs through the volumes in their thousands.
    Outside a span neither can be anything but the mark, since prose has no
    operators in it.
    """
    math = in_math(text)
    rs = list(text)
    out: list[str] = []
    n = 0
    for i, ch in enumerate(rs):
        bad = ch in ORNAMENTS
        if (bad and not math[i]) or _bare_star(rs, i, math):
            out.append(STAR)
            n += 1
            continue
        out.append(ch)
    return "".join(out), n


# The same substitution as the braced list below, without the braces. A single
# letter argument does not need them and a model does not always write them: the
# first live page came back with `$\mathbb Z$`, which the braced list missed.
# The following character must not be a letter, or `\mathbb Zeta` loses its tail.
_BARE_BLACKBOARD = re.compile(r"\\mathbb\s+([ZQRCNFP])(?:\{\})?\b", re.A)

# Only substitutions that are unambiguously wrong in this corpus are made. The
# minus sign, the two dash lengths and the quotation marks all carry meaning in
# mathematics and are left alone.
_SUBSTITUTIONS = (
    # A model told to write bold sets sometimes writes blackboard bold anyway.
    # Bourbaki prints bold, and a corpus that mixes the two makes every search
    # for a ring name miss half its hits.
    ("\\mathbb{Z}", "\\mathbf{Z}"),
    ("\\mathbb{Q}", "\\mathbf{Q}"),
    ("\\mathbb{R}", "\\mathbf{R}"),
    ("\\mathbb{C}", "\\mathbf{C}"),
    ("\\mathbb{N}", "\\mathbf{N}"),
    ("\\mathbb{F}", "\\mathbf{F}"),
    ("\\mathbb{P}", "\\mathbf{P}"),
    # The dangerous bend comes back as several near misses.
    ("\u26a0", "\u2621"),
    ("\u26f0", "\u2621"),
    # Non breaking and thin spaces read as ordinary spaces everywhere they
    # appear in these volumes, and they break every word match if kept.
    ("\u00a0", " "),
    ("\u2009", " "),
    ("\u202f", " "),
    # Zero width characters carry nothing and hide differences in a diff.
    ("\u200b", ""),
    ("\ufeff", ""),
)


def _substitute(text: str) -> str:
    text = _BARE_BLACKBOARD.sub(r"\\mathbf{\1}", text)
    text, _ = dollars(text)
    for old, new in _SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


def _trim_right(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalise(text: str) -> str:
    """Fix the typography a model substitutes for what the page prints.

    The star is not in the substitution table with the dangerous bend, though
    the two faults are the same fault. A table cannot see where the mathematics
    is, and two of the five spellings the star comes back as are operations
    inside a span. So `stars` runs on its own and after the delimiters have been
    turned round, since it has to be able to find the spans to keep out of them.

    Trailing space goes from every line, which is invisible in review and shows
    up in every later diff.
    """
    text, _ = stars(_substitute(text))
    return _trim_right(text)


def normalise_prose(text: str) -> str:
    """`normalise` with the star left out, for text somebody wrote by hand.

    Markdown uses the asterisk for its own purposes: a bullet list opens with
    one at the head of a line with a space after it, which is exactly the shape
    `_bare_star` was written to find. On a scanned page that shape is Bourbaki's
    mark, because the volumes set no bullet lists. In a solution it is a list,
    and turning it into `\\*` puts a backslash at the head of every item.
    """
    return _trim_right(_substitute(text))
