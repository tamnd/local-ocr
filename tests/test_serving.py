"""The shortlist and the command line it turns into.

None of this needs a GPU, which is the reason it is worth having. The command
line that starts a reader is the difference between a run that can be repeated
and one that cannot, and it is not the sort of thing to find out about by
reading a systemd journal on a machine in another room.
"""

from __future__ import annotations

import json

import pytest

from local_ocr import serving

SAMPLE = """
[reader-a]
repo = "allenai/olmOCR-2-7B-1025-FP8"
revision = "40bd7202494b8264ee17ada08b401b5aab7a9ce1"
port = 8801
what = "the default"
args = ["--max-model-len", "16384", "--gpu-memory-utilization", "0.86"]

[reader-z]
repo = "somewhere/something"
"""


@pytest.fixture
def shortlist(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class TestLoad:
    def test_every_entry_is_read(self, shortlist):
        entries = serving.load(shortlist)
        assert sorted(entries) == ["reader-a", "reader-z"]

    def test_an_entry_that_names_no_port_gets_one_that_is_not_vllms_own(self, shortlist):
        # 8000 would collide with a hand started server on the same box, and the
        # collision is silent: requests go somewhere, just not where you meant.
        assert serving.load(shortlist)["reader-z"].port == serving.DEFAULT_PORT
        assert serving.DEFAULT_PORT != 8000

    def test_an_entry_that_names_no_revision_is_on_main_and_says_so(self, shortlist):
        entry = serving.load(shortlist)["reader-z"]
        assert entry.revision == "main"
        assert not entry.pinned

    def test_a_pinned_entry_knows_it(self, shortlist):
        assert serving.load(shortlist)["reader-a"].pinned


class TestCommand:
    def test_the_revision_is_always_passed(self, shortlist):
        # Including the unpinned one. `--revision main` resolves to whatever main
        # is today, but it puts the word in the process listing and in the unit
        # file, which is where somebody will look when two runs disagree.
        for entry in serving.load(shortlist).values():
            command = entry.command()
            assert "--revision" in command
            assert command[command.index("--revision") + 1] == entry.revision

    def test_the_served_name_is_the_entry_and_not_the_repository(self, shortlist):
        command = serving.load(shortlist)["reader-a"].command()
        assert command[command.index("--served-model-name") + 1] == "reader-a"
        assert command[2] == "allenai/olmOCR-2-7B-1025-FP8"

    def test_the_entrys_own_flags_come_last(self, shortlist):
        # So that an entry can override anything this module puts in front of
        # it, vLLM taking the later of two of the same flag.
        entry = serving.load(shortlist)["reader-a"]
        assert entry.command()[-len(entry.args) :] == list(entry.args)

    def test_the_binary_can_be_a_path_into_a_venv(self, shortlist):
        # On gamingpc it is, because a systemd unit has no PATH worth relying on.
        command = serving.load(shortlist)["reader-a"].command(
            "/home/gopher/local-ocr/.venv/bin/vllm"
        )
        assert command[0] == "/home/gopher/local-ocr/.venv/bin/vllm"
        assert command[1] == "serve"

    def test_the_shell_form_is_the_same_command(self, shortlist):
        entry = serving.load(shortlist)["reader-a"]
        assert entry.shell().split() == entry.command()

    def test_the_url_is_loopback(self, shortlist):
        # The reader is reached over an ssh tunnel or from the batch process on
        # the same machine. It has no authentication, so it does not get an
        # address that a network can find.
        assert serving.load(shortlist)["reader-a"].url == "http://127.0.0.1:8801/v1"


class TestModel:
    def test_a_name_that_is_not_there_says_which_names_are(self, shortlist):
        with pytest.raises(serving.NoSuchModel) as caught:
            serving.model("reader-q", shortlist)
        assert "reader-a" in str(caught.value)
        assert "reader-z" in str(caught.value)


class TestTheShippedShortlist:
    """The real models.toml, because a typo in it breaks the machine and not a test."""

    def test_it_parses(self):
        entries = serving.load()
        assert "reader-a" in entries

    def test_every_reader_is_pinned(self):
        # Not just the default. dots.ocr moved repository between the survey and
        # the bake off, from rednote-hilab to dots-studio, which is what a branch
        # name is worth as a record of what was served.
        for entry in serving.load().values():
            assert entry.pinned, entry.name

    def test_a_revision_is_a_full_commit_and_not_a_short_one(self):
        # A short sha is ambiguous in principle and, more to the point, it is
        # what somebody types from memory. These are copied from the hub API.
        for entry in serving.load().values():
            assert len(entry.revision) == 40, entry.name

    def test_no_two_readers_share_a_port(self):
        # All three are meant to be able to sit on the card at once for the bake
        # off, and a shared port would show up as one of them silently answering
        # for another.
        ports = [entry.port for entry in serving.load().values()]
        assert len(ports) == len(set(ports))

    def test_every_reader_is_asked_for_one_page_at_a_time(self):
        # The queue, the acceptance rules and the page file are all per page, and
        # two pages in one prompt would let a single bad answer poison both.
        #
        # The value is JSON and not `image=1`, which is what vLLM 0.27 accepts
        # and what the first attempt to start reader-a under systemd got wrong:
        # "Value image=1 cannot be converted", a restart loop, and no reader.
        for entry in serving.load().values():
            flags = list(entry.args)
            limit = flags[flags.index("--limit-mm-per-prompt") + 1]
            assert json.loads(limit) == {"image": 1}, entry.name

    def test_no_reader_is_served_at_the_setting_that_trims_pages(self):
        # These are solo shares: one reader on the card at a time, which is what
        # M1 and the M4 bake off do. M6 puts two on it at once and lowers them,
        # and that is an edit to this file rather than a flag somewhere else.
        #
        # 0.9 and above has been reported to trim outputs under load, because the
        # vision encoder's activations are not in vLLM's KV budget and a 300 dpi
        # page is a large one. A trimmed page fails the short rule and is read
        # again, so the aggressive setting costs throughput rather than gaining
        # it.
        for entry in serving.load().values():
            flags = list(entry.args)
            share = float(flags[flags.index("--gpu-memory-utilization") + 1])
            assert 0.0 < share < 0.9, entry.name

    def test_no_reader_is_served_with_an_eight_bit_kv_cache(self):
        # It is the obvious saving on a 24 GB card and it was in the spec's
        # starting configuration. It also turned the first page ever read on this
        # machine into "2.5.1.1." repeated four thousand times at temperature
        # zero, and the same page read correctly the moment it came out. The flag
        # is cheap to add back by accident, so it is expensive to add back here.
        for entry in serving.load().values():
            assert "--kv-cache-dtype" not in entry.args, entry.name

    def test_every_reader_says_what_it_is_for(self):
        for entry in serving.load().values():
            assert len(entry.what) > 80, entry.name
