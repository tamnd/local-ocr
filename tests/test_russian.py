"""The two Russian gates.

What is being tested is the distinction each gate draws, not that it fires. A
gate that flags every unknown word is a gate whose output nobody reads, and a
gate that flags every Cyrillic character in a formula would bury the thirty
lookalike pairs, which are the ones a person cannot find by looking.

The vocabularies here are small and written out, because a test that builds one
from a corpus is a test that depends on the corpus and fails when somebody adds
an issue to it.
"""

from __future__ import annotations

from pathlib import Path

from local_ocr.russian import (
    Flag,
    carried,
    fold,
    gates,
    homoglyphs,
    lexicon,
    oov,
    prose,
    rate,
    vocabulary,
)

# Enough words to say what is in vocabulary and what is not, in the register
# these pages are written in.
WORDS = {
    "теорема",
    "доказательство",
    "точка",
    "точки",
    "прямая",
    "плоскость",
    "число",
    "числа",
    "какой-нибудь",
    "треугольник",
    "окружность",
    "равна",
    "если",
    "тогда",
    "и",
}


# ---------------------------------------------------------------------------
# Looking a word up


def test_yo_and_ye_are_the_same_word():
    """Print writes ё where a reader needs it and omits it everywhere else.

    Both spellings are in every real corpus, so a vocabulary that keeps them
    apart flags whichever one the page happened not to use, which is a fact
    about typography and not about the reading.
    """
    assert fold("Ещё") == fold("еще") == "еще"


def test_case_is_not_a_difference():
    assert fold("ТЕОРЕМА") == fold("Теорема") == "теорема"


# ---------------------------------------------------------------------------
# The vocabulary gate


def test_a_word_repeated_wrong_is_the_finding():
    """The failure Kvant measured: one word off by a letter, down the page."""
    body = "Теорема о точке. Точка лежит на трямой. Через трямая проходит трямая."
    flags = oov(body, WORDS)
    assert [f.detail for f in flags] == ["трямая"]
    assert flags[0].count == 2
    assert flags[0].kind == "oov"


def test_one_unknown_word_is_not_a_finding():
    """A surname, a term of art, or a word this vocabulary happens to lack.

    Flagging these would put a flag on every page and the list would stop being
    read, which costs more than the one word it would have caught.
    """
    body = "Теорема Понтрягина о точке. Точка лежит на прямой, и прямая одна."
    assert oov(body, WORDS) == []


def test_short_tokens_are_left_alone():
    """One and two letter Cyrillic tokens are prepositions and stray italics.

    A two letter token repeated is not a finding, it is Russian, or it is one
    italic variable that the reading brought out of a formula into the prose.
    Either way the gate has nothing useful to say about it.
    """
    body = "Прямая ху, и прямая ху, и прямая."
    assert [f.detail for f in oov(body, WORDS)] == []


def test_a_hyphenated_compound_is_one_word():
    """Splitting on the hyphen would put both halves in vocabulary and pass.

    So a wrong hyphenation would be exactly the case the gate misses.
    """
    good = "Возьмём какой-нибудь треугольник. Какой-нибудь треугольник подойдёт."
    assert oov(good, WORDS) == []
    bad = good.replace("какой-нибудь", "какой-нибуть").replace("Какой-нибудь", "Какой-нибуть")
    assert [f.detail for f in oov(bad, WORDS)] == ["какой-нибуть"]


def test_the_worst_offender_is_first_and_the_order_is_stable():
    body = "Прямая аабб и прямая аабб и прямая аабб. Плоскость ввгг и плоскость ввгг."
    flags = oov(body, WORDS)
    assert [(f.detail, f.count) for f in flags] == [("аабб", 3), ("ввгг", 2)]
    assert oov(body, WORDS) == flags


def test_an_empty_vocabulary_says_nothing_rather_than_everything():
    """A gate that has not been given a vocabulary has no opinion.

    The alternative is that a missing build step turns every page on the run
    into a flagged page, which reads as a catastrophic run rather than as a
    missing file.
    """
    assert oov("Теорема о трямой трямой.", set()) == []


# ---------------------------------------------------------------------------
# Mathematics is not prose


def test_the_gate_does_not_read_the_formulae():
    """A formula is single letters and command names and none of it is Russian.

    Left in, every page with mathematics on it would flag on its own notation.
    """
    body = "Точка $x$ на прямой $\\alpha\\beta$, и точка $y$ тоже, и $\\alpha\\beta$."
    assert oov(body, WORDS) == []


def test_display_mathematics_is_cut_out_too():
    body = "Число равна\n$$\n\\sum_{n=0}^\\infty q^n = \\frac{1}{1-q}\n$$\nи число равна.\n"
    assert oov(body, WORDS) == []


def test_prose_keeps_the_words_around_a_formula():
    kept = prose("Точка $x$ лежит на прямой $y$ всегда.")
    assert "Точка" in kept
    assert "лежит" in kept
    assert "всегда" in kept
    assert "x" not in kept


def test_prose_leaves_a_page_with_no_mathematics_alone():
    body = "Теорема о точке и прямой."
    assert prose(body) == body


# ---------------------------------------------------------------------------
# Cyrillic inside mathematics


def test_a_cyrillic_lookalike_in_a_formula_is_named_with_what_it_looks_like():
    """A Cyrillic er where a Latin p belongs. The two draw the same picture.

    Nobody finds this by reading the page, which is the reason the flag says
    which Latin letter it is being mistaken for.
    """
    body = "Плотность $\u0440 = m/V$ равна плотности $\u0440$."
    flags = homoglyphs(body)
    assert [f.kind for f in flags] == ["homoglyph"]
    assert "reads as p" in flags[0].detail
    assert flags[0].count == 2


def test_a_cyrillic_letter_with_no_latin_lookalike_is_left_alone():
    """A Cyrillic ж is not mistakable for anything Latin.

    Anyone reading the page sees it, so reporting it spends a flag on the one
    case a person can already find. Reporting the whole alphabet is what the
    first version of this gate did and it fired on a fifth of all pages.
    """
    assert homoglyphs("Пусть $ж + 1$ равно.") == []


def test_a_russian_unit_is_not_a_finding():
    """Units in a Russian paper are written in Russian, inside \\text.

    Leaving these in was three quarters of the noise: 19.5 per cent of pages
    down to 5.2 on the same 400 once the \\text groups came out.
    """
    body = "$R = 8.31 \\text{ Дж/(моль·К)}$"
    assert homoglyphs(body) == []


def test_a_multi_letter_subscript_is_an_abbreviation_and_not_a_variable():
    """`F_{тр}` is the force of friction, spelled the way Russian spells it.

    Length is what separates the two: one letter is a variable and is what gets
    mistaken for a Latin one, two or more is a word cut short.
    """
    assert homoglyphs("$F_{тр} = \\mu N$") == []


def test_a_russian_single_letter_subscript_is_not_a_finding_either():
    """`v_н` is the initial velocity, and н has no Latin lookalike."""
    assert homoglyphs("$v_н = 0$") == []


def test_an_initial_in_a_signature_is_not_a_variable():
    """A single Cyrillic letter followed by a stop is an initial.

    Two of the ten characters the gate still returned over 400 clean pages, and
    both of them came from one signature.
    """
    assert homoglyphs("$^{А.Н.Колмогоров.}$") == []


def test_the_worst_offender_is_first_among_the_lookalikes():
    body = "$с_{1} d с_{2}$ и $а + с_{3}$"
    assert [(f.detail, f.count) for f in homoglyphs(body)] == [
        ("с reads as c", 3),
        ("а reads as a", 1),
    ]


def test_cyrillic_in_the_prose_is_not_a_finding():
    """It is a Russian page. The gate is about the formulae and only those."""
    assert homoglyphs("Плотность тела равна отношению массы к объёму.") == []


def test_latin_in_a_formula_is_what_is_expected():
    assert homoglyphs("Пусть $p = m/V$ и $q = 1$.") == []


def test_an_unclosed_formula_does_not_swallow_the_page():
    """The open span is not returned as a span, so nothing is scanned twice.

    A page with an unclosed dollar fails the math rule and never reaches here,
    but reaching here should not then report the whole rest of the page.
    """
    assert homoglyphs("Пусть $\u0440 = 1 и дальше идёт обычный русский текст.") == []


# ---------------------------------------------------------------------------
# Both at once, and the line that goes in the issue


def test_both_gates_run_on_the_same_page():
    """They come from different failures of one reading, so neither preempts.

    Stopping at the first would hide whichever is second, and the one that is
    hard to find by eye is as likely to be second as first.
    """
    body = "Трямая \u0440 в точке. Трямая $\u0440 = 1$ и трямая."
    kinds = {f.kind for f in gates(body, WORDS)}
    assert kinds == {"oov", "homoglyph"}


def test_the_summary_counts_pages_and_not_only_flags():
    pages = {
        "0001.md": [Flag("oov", "трямая", 3)],
        "0002.md": [],
        "0003.md": [Flag("homoglyph", "\u0440 reads as p", 1), Flag("oov", "ввгг", 2)],
    }
    line = rate(pages)
    assert "2 of 3 pages flagged" in line
    assert "66.7 %" in line
    assert "1 homoglyph" in line
    assert "2 oov" in line


def test_a_clean_run_says_so_without_a_kind_list():
    assert rate({"0001.md": [], "0002.md": []}) == "0 of 2 pages flagged, 0.0 %"


def test_an_empty_run_is_not_a_division_by_zero():
    assert rate({}) == "no pages"


def test_a_flag_prints_its_count_only_when_there_is_one_to_print():
    assert str(Flag("oov", "трямая", 1)) == "oov: трямая"
    assert str(Flag("oov", "трямая", 4)) == "oov: трямая (4 times)"


# ---------------------------------------------------------------------------
# Building the vocabulary


def test_the_vocabulary_drops_what_the_corpus_barely_says(tmp_path: Path):
    """A word seen once may be a typo in the source, and laundering it into the
    vocabulary would teach the gate to accept exactly the error it looks for.
    """
    (tmp_path / "a.md").write_text("точка точка точка прямая\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("точка прямая трямая\n", encoding="utf-8")
    words = vocabulary(tmp_path, seen=3)
    assert "точка" in words
    assert "прямая" not in words
    assert "трямая" not in words


def test_the_vocabulary_is_built_folded(tmp_path: Path):
    (tmp_path / "a.md").write_text("Ещё ещё еще\n", encoding="utf-8")
    assert vocabulary(tmp_path, seen=3) == {"еще"}


def test_the_vocabulary_reads_the_whole_tree(tmp_path: Path):
    deep = tmp_path / "1975" / "03" / "articles"
    deep.mkdir(parents=True)
    (deep / "one.md").write_text("окружность окружность окружность\n", encoding="utf-8")
    assert "окружность" in vocabulary(tmp_path)


# ---------------------------------------------------------------------------
# The vocabulary the corpus already has written down


def test_the_lexicon_is_read_folded_so_a_lookup_never_has_to_fold_it(tmp_path: Path):
    """1.5 M entries folded once beats folding the page's words against them.

    And it is the same fold `oov` uses, so a list that spells a word with ё and
    a page that spells it without are the same entry rather than an unknown
    word repeated down the page, which is exactly the flag this gate exists to
    make mean something.
    """
    path = tmp_path / "lexicon.txt"
    path.write_text("Ещё\nТОЧКА\nпрямая\n", encoding="utf-8")
    assert lexicon(path) == {"еще", "точка", "прямая"}


def test_the_header_the_go_side_writes_is_not_taken_for_vocabulary(tmp_path: Path):
    """The real file opens with several comment lines saying where it came from."""
    path = tmp_path / "lexicon.txt"
    path.write_text("# built from the corpus\n# plus a published form list\nточка\n", "utf-8")
    assert lexicon(path) == {"точка"}


def test_a_blank_line_in_the_lexicon_is_not_a_word(tmp_path: Path):
    path = tmp_path / "lexicon.txt"
    path.write_text("точка\n\n   \nпрямая\n", encoding="utf-8")
    assert lexicon(path) == {"точка", "прямая"}


def test_the_lexicon_and_the_gate_agree_on_a_page(tmp_path: Path):
    """The pair that matters, since either alone can be right and the two wrong."""
    path = tmp_path / "lexicon.txt"
    path.write_text("точка\nпрямая\n", encoding="utf-8")
    words = lexicon(path)
    assert oov("точка прямая точка", words) == []
    assert [f.detail for f in oov("точка кривоая кривоая", words)] == ["кривоая"]


# ---------------------------------------------------------------------------
# Compounds
#
# The vocabulary the gate really runs against holds 1 557 248 word forms and not
# one of them has a hyphen in it, so a compound has to be judged by its parts or
# every compound on every page is unknown.

KNOWN = {"из", "за", "что", "то", "какой", "бета", "распада", "морскую", "кастрюле"}


def test_a_compound_the_list_does_not_hold_is_known_when_its_parts_are():
    assert carried("из-за", KNOWN)
    assert carried("что-то", KNOWN)
    assert carried("бета-распада", KNOWN)


def test_a_compound_with_one_part_the_list_does_not_hold_is_not_known():
    """The stricter reading. A misread inside a compound leaves a bad part."""
    assert not carried("кастрюле-скороварке", KNOWN)


def test_a_plain_word_is_still_judged_whole():
    assert carried("морскую", KNOWN)
    assert not carried("морьскую", KNOWN)


def test_a_word_the_list_holds_whole_is_known_even_with_a_hyphen_in_it():
    assert carried("из-за", KNOWN | {"из-за"})


def test_a_bare_hyphen_is_not_a_word():
    assert not carried("-", KNOWN)
    assert not carried("--", KNOWN)


def test_the_gate_stops_flagging_ordinary_compounds():
    """What this cost on real pages: 43 of 57 flags on kvant-dev were these."""
    page = "из-за из-за что-то что-то морьскую морьскую"
    assert [f.detail for f in oov(page, KNOWN)] == ["морьскую"]
