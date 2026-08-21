"""Two gates for Russian pages, and the vocabulary one of them reads.

The eight rules in `local_ocr.rules` are a transcription of the Go rules and
they are about Bourbaki: balanced dollars, a running head, a page label that
agrees with the page map. None of them says anything about the language, which
is fine for French and English and not fine for Russian, because the two ways a
Russian page goes wrong are both invisible to a rule that only counts
delimiters.

The first is a word the model renders wrong the same way every time it appears.
Kvant measured this: a single word off by a letter, repeated down the page,
reading as ordinary Russian to anything that is not checking the spelling. One
occurrence of an unknown word is a proper noun or a term of art and is not
worth a person's time. The same unknown word twice on one page is a systematic
error, and that is what `oov` looks for.

The second is Cyrillic inside mathematics. Russian mathematical typesetting
does put the occasional Cyrillic letter in a formula, but what actually happens
on these pages is a Latin variable coming back as its Cyrillic lookalike. There
are thirty odd pairs that render identically in most fonts, so `$\\rho$` and a
Cyrillic er are the same picture and different characters, and the difference
survives into every downstream consumer that ever compares two strings. That is
what `homoglyphs` looks for.

Neither gate rejects a page on its own. They flag, and the flag carries the
token or the character so somebody can look, because a rate is not a finding.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from local_ocr.rules.mathtex import split

WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
"""A Russian word, hyphenated compounds included.

Hyphens are joined rather than split because the compounds are the interesting
half: `какой-нибудь` is one entry in any vocabulary and two tokens that are both
in it, so splitting would let a wrong hyphen through.
"""

SHORTEST = 3
"""The shortest token the vocabulary gate has an opinion about.

One and two letter Cyrillic tokens are prepositions and conjunctions, and they
are also what a single italic variable in prose comes back as. Below three
letters the gate would spend all its flags on `в` and `и` and on the letters of
formulae that were never in a formula.
"""

REPEATED = 2
"""How many times an unknown word has to appear before it is worth reporting.

The whole distinction the gate exists to draw. Once is a proper noun, a surname,
a transliterated foreign term, or one of the several thousand words a magazine
uses that a corpus vocabulary happens not to carry. Twice is the model doing the
same thing to the same word, which is the failure Kvant measured.
"""

SEEN = 3
"""How often a word has to occur in the source corpus to count as vocabulary.

A vocabulary built from text that was itself read by a model would launder that
model's errors into the gate, so the source is the born digital text and not any
page this repository has produced. Even there a word that appears once may be a
typo in the original, and at three occurrences across ten million tokens the
list is 115 605 words and the hapax noise is gone.
"""

CYRILLIC = re.compile(r"[Ѐ-ӿ]+")
"""A run of Cyrillic, taken whole because its length is the signal.

One letter in a formula is a variable and is the thing that gets mistaken for a
Latin one. Two or more is an abbreviation of a Russian word, which is how the
subscripts in a Russian formula are written and is not an error.
"""

TEXTY = re.compile(r"\\(?:text|textrm|mathrm|mbox|operatorname|textbf|textit)\s*\{[^{}]*\}")
"""The groups inside a formula that hold words rather than mathematics.

This is where the units go, and units in a Russian paper are written in Russian.
Leaving these in was three quarters of the noise the first version of the
Cyrillic gate produced.
"""

HOMOGLYPHS = {
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
    "Ѕ": "S",
    "І": "I",
    "Ј": "J",
    "Ԛ": "Q",
    "Ԝ": "W",
    "Ғ": "F",
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "ѕ": "s",
    "і": "i",
    "ј": "j",
    "ԛ": "q",
    "ԝ": "w",
    "ѐ": "e",
    "ё": "e",
}
"""Cyrillic letters that draw the same picture as a Latin letter.

Why these are worth naming apart from the rest of the alphabet: a Cyrillic ж in
a formula is visibly Cyrillic and somebody reading the page will see it. A
Cyrillic er in place of a Latin p is not visible in any font this corpus is set
in, it compares unequal to the p everywhere downstream, and it will sit in the
text until something looks at the code points.
"""


@dataclass(frozen=True)
class Flag:
    """One thing on a page worth a person's attention.

    Not a `rules.Problem`. Those carry a `Rule` from the Go enum and mean the
    page was not accepted; these mean the page passed everything and may still
    be wrong, which is a different sentence and deserves a different type.
    """

    kind: str
    detail: str
    count: int = 1

    def __str__(self) -> str:
        if self.count > 1:
            return f"{self.kind}: {self.detail} ({self.count} times)"
        return f"{self.kind}: {self.detail}"


def fold(word: str) -> str:
    """The form a word is looked up under.

    Lowercased, and ё folded to е. The fold is not cosmetic: Russian printing
    writes ё where it is needed for a reader and omits it everywhere else, the
    same word appears both ways in any real corpus, and a vocabulary that keeps
    them apart flags the spelling the page happened not to use.
    """
    return word.lower().replace("ё", "е")


def vocabulary(root: Path, *, seen: int = SEEN) -> set[str]:
    """Build the vocabulary from a tree of Markdown.

    Point this at born digital text. The Kvant Russian tree is 22 384 files and
    ten million tokens and takes a few seconds, which is why the result is meant
    to be built once and written down rather than computed per page.

    Word forms and not lemmas, which is why the corpus has to be this size.
    Russian inflects a noun through a dozen forms and a verb through more, and
    the gate looks up whichever form the page prints. A small vocabulary would
    hold `точка` and not `точке` and would flag the second as an unknown word,
    so the thing that makes this work is that ten million tokens of the same
    magazine contain the forms the magazine uses.
    """
    counts: Counter[str] = Counter()
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        counts.update(fold(w) for w in WORD.findall(text))
    return {word for word, n in counts.items() if n >= seen}


def prose(body: str) -> str:
    """The body with its mathematics removed.

    The vocabulary gate reads this and not the whole page. A formula is full of
    single letters and command names and none of it is Russian, so leaving it in
    would put the gate's flags where its opinion is worthless.
    """
    spans, _ = split(body)
    if not spans:
        return body
    runes = list(body)
    keep = [True] * len(runes)
    for span in spans:
        for i in range(max(0, span.start), min(len(runes), span.end)):
            keep[i] = False
    return "".join(r for r, k in zip(runes, keep, strict=True) if k)


def oov(body: str, words: set[str], *, repeated: int = REPEATED) -> list[Flag]:
    """Words the vocabulary does not carry, that the page uses more than once.

    Sorted by count and then alphabetically, so that the worst offender on a
    page is the first thing read and so that two runs over the same page give
    the same list in the same order.
    """
    if not words:
        return []
    counts: Counter[str] = Counter()
    for raw in WORD.findall(prose(body)):
        word = fold(raw)
        if len(word.replace("-", "")) >= SHORTEST and word not in words:
            counts[word] += 1
    flags = [Flag("oov", word, n) for word, n in counts.items() if n >= repeated]
    return sorted(flags, key=lambda f: (-f.count, f.detail))


def homoglyphs(body: str) -> list[Flag]:
    """A lone Cyrillic lookalike standing where a Latin variable belongs.

    This started as the obvious gate, which was every Cyrillic character in
    every math span, on the reasoning that a formula is not where Russian
    belongs. Over 400 born digital Kvant pages that fired on 19.5 per cent of
    them, and reading what it caught says the reasoning is just wrong about
    Russian. A Russian formula carries Russian in three ordinary places: the
    units, written `\\text{Дж/(моль·К)}`; multi letter subscripts that abbreviate
    a Russian word, so `F_{тр}` for friction; and single letter subscripts doing
    the same, so `v_н` for the initial value. None of those is an error and
    together they are almost all of it.

    So three exclusions, each measured on the same 400 pages. Dropping the
    `\\text` groups takes it from 19.5 per cent to 5.2. Reporting only the
    lookalikes and not the whole alphabet, since a Cyrillic ж is visible to
    anyone reading the page and a Cyrillic er is not, takes it to a smaller
    number again. Requiring the run to be one letter long, which is what
    separates a variable from an abbreviation, takes it to 1.0 per cent, and a
    single letter followed by a stop is an initial rather than a variable.

    What survives is worth reading. Of the ten characters left on those four
    pages, five are `с_{1} d с_{2} d ... с_{20}` where the sequence is plainly
    Latin c, one is `с^{2}`, two are a garbled superscript, and two are the
    initials in a signature that the stop rule now takes out as well.
    """
    spans, _ = split(body)
    found: Counter[str] = Counter()
    for span in spans:
        text = TEXTY.sub(" ", span.text)
        for match in CYRILLIC.finditer(text):
            run = match.group()
            if len(run) != 1 or run not in HOMOGLYPHS:
                continue
            after = text[match.end() : match.end() + 1]
            if after == ".":
                continue
            found[run] += 1
    flags = [Flag("homoglyph", f"{ch} reads as {HOMOGLYPHS[ch]}", n) for ch, n in found.items()]
    return sorted(flags, key=lambda f: (-f.count, f.detail))


def gates(body: str, words: set[str]) -> list[Flag]:
    """Both gates over one page.

    Both always run. A page can carry a repeated misspelling and a Cyrillic er
    at once, they come from different failures of the same reading, and stopping
    at the first would hide the one that is harder to find by eye.
    """
    return oov(body, words) + homoglyphs(body)


def rate(pages: dict[str, list[Flag]]) -> str:
    """One line saying how a run went, for the milestone issue.

    A rate on its own is the thing this module exists not to produce, so the
    line carries the count of pages and the count of flags and leaves the flags
    themselves to whoever prints them.
    """
    flagged = sum(1 for flags in pages.values() if flags)
    total = len(pages)
    if not total:
        return "no pages"
    counts = Counter(f.kind for flags in pages.values() for f in flags)
    kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
    share = 100 * flagged / total
    return f"{flagged} of {total} pages flagged, {share:.1f} %" + (f": {kinds}" if kinds else "")
