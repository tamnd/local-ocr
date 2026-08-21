"""The recipe, the dataset, and the run card. Nothing here touches a card.

That is the point of the split: what decides whether a training run is a valid
one is checked on any machine, in CI, before anybody books the GPU for a night.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_ocr import finetune, golden, pool

PROMPT = "Read the page. Running head first. \\mathbf{Z} and never \\mathbb{Z}.\n"


def example(page_id: str = "alg-viii/9001", split: str = "train") -> pool.Example:
    return pool.Example(
        id=page_id,
        book=page_id.split("/")[0],
        image=f"/corpus/images/{page_id}.png",
        completion="THE RADICAL A VIII.13\n\n## § 2. THE RADICAL\n",
        split=split,
    )


def recipe(**kwargs) -> finetune.Recipe:
    settings = {"base": "reader-a", "revision": "b5c3f0a"}
    settings.update(kwargs)
    return finetune.Recipe(**settings)


class TestRecipe:
    def test_the_vision_tower_is_refused(self):
        """The setting §08 says costs a week. vLLM cannot serve the adapter."""
        with pytest.raises(finetune.Refused) as err:
            finetune.check(recipe(finetune_vision_layers=True))
        assert "vLLM" in str(err.value)

    def test_an_unpinned_base_is_refused(self):
        for revision in ("", "main"):
            with pytest.raises(finetune.Refused):
                finetune.check(recipe(revision=revision))

    def test_a_nonzero_temperature_is_refused(self):
        """Two checkpoints have to differ by their weights and by nothing else."""
        with pytest.raises(finetune.Refused):
            finetune.check(recipe(temperature=0.2))

    def test_the_default_recipe_is_the_one_the_spec_names(self):
        one = recipe()
        finetune.check(one)
        assert one.peft_kwargs() == {
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "r": 16,
            "lora_alpha": 16,
        }
        assert one.completion_only, "the prompt is 1400 identical tokens an example"


class TestDataset:
    def test_the_prompt_is_attached_here_and_not_in_the_pool(self):
        one = finetune.conversation(example(), PROMPT)
        user = one["messages"][0]["content"]
        assert user[0] == {"type": "image", "image": "/corpus/images/alg-viii/9001.png"}
        assert user[1] == {"type": "text", "text": PROMPT}

    def test_the_completion_is_what_the_model_is_trained_to_say(self):
        one = finetune.conversation(example(), PROMPT)
        answer = one["messages"][1]
        assert answer["role"] == "assistant"
        assert answer["content"][0]["text"].startswith("THE RADICAL A VIII.13")

    def test_the_image_goes_in_by_path(self):
        """2500 pages of 300 dpi PNG inline is gigabytes of unreadable base64."""
        line = finetune.dataset([example()], PROMPT)
        assert "/corpus/images/alg-viii/9001.png" in line
        assert "base64" not in line

    def test_each_side_of_the_split_is_written_on_its_own(self):
        both = [example("alg-viii/9001"), example("alg-viii/9002", split="val")]
        assert len(finetune.dataset(both, PROMPT, "train").splitlines()) == 1
        assert len(finetune.dataset(both, PROMPT, "val").splitlines()) == 1

    def test_a_split_with_nothing_in_it_is_refused(self):
        with pytest.raises(finetune.Refused):
            finetune.dataset([example()], PROMPT, "val")


class TestReadingThePoolBack:
    def test_a_pool_file_is_not_taken_on_trust(self, tmp_path: Path):
        """It was written by another machine weeks ago and can be edited by hand."""
        page_id = golden.read_manifest("golden-test")[0]
        path = tmp_path / "pool.jsonl"
        path.write_text(
            pool.to_jsonl(pool.Pool(examples=[example(), example(page_id)])),
            encoding="utf-8",
        )
        with pytest.raises(pool.Contaminated) as err:
            finetune.read_pool(path)
        assert page_id in str(err.value)

    def test_a_line_that_is_not_a_pool_line_says_which_line(self, tmp_path: Path):
        path = tmp_path / "pool.jsonl"
        path.write_text('{"id": "alg-viii/9001"}\n', encoding="utf-8")
        with pytest.raises(finetune.Refused) as err:
            finetune.read_pool(path)
        assert ":1" in str(err.value)

    def test_a_clean_pool_reads_back_as_it_was_written(self, tmp_path: Path):
        path = tmp_path / "pool.jsonl"
        written = [example("alg-viii/9001"), example("alg-viii/9002", split="val")]
        path.write_text(pool.to_jsonl(pool.Pool(examples=written)), encoding="utf-8")
        assert finetune.read_pool(path) == written


class TestCard:
    def test_it_records_the_prompt_by_digest_and_not_by_text(self):
        run = finetune.plan([example()], recipe(), PROMPT)
        card = run.card()
        assert finetune.digest(PROMPT) in card
        assert "\\mathbb{Z}" not in card

    def test_it_says_the_vision_tower_was_left_alone_and_why(self):
        card = finetune.plan([example()], recipe(), PROMPT).card()
        assert "`finetune_vision_layers` | `False`" in card
        assert "does not serve LoRA adapters on vision layers" in card

    def test_it_counts_the_dev_pages_it_trained_on(self):
        """§08 allows training on golden-dev. It does not allow doing it quietly."""
        page_id = golden.read_manifest("golden-dev")[0]
        run = finetune.plan([example(page_id), example("alg-viii/9001")], recipe(), PROMPT)
        assert run.dev_pages == 1
        assert "1 of the training pages are `golden-dev` pages" in run.card()

    def test_a_recipe_that_cannot_be_served_never_reaches_a_card(self):
        with pytest.raises(finetune.Refused):
            finetune.plan([example()], recipe(finetune_vision_layers=True), PROMPT)


class TestCommand:
    def test_it_writes_both_splits_and_the_card(self, tmp_path: Path):
        from local_ocr import cli

        pool_file = tmp_path / "pool.jsonl"
        pool_file.write_text(
            pool.to_jsonl(
                pool.Pool(examples=[example("alg-viii/9001"), example("alg-viii/9002", "val")])
            ),
            encoding="utf-8",
        )
        prompt = tmp_path / "prompt.txt"
        prompt.write_text(PROMPT, encoding="utf-8")
        out = tmp_path / "run"
        code = cli.main(
            [
                "finetune",
                "--pool",
                str(pool_file),
                "--prompt-file",
                str(prompt),
                "--base",
                "reader-a",
                "--revision",
                "b5c3f0a",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        assert len((out / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        assert len((out / "val.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        assert "# Training run" in (out / "run.md").read_text(encoding="utf-8")
        first = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert first["id"] == "alg-viii/9001"

    def test_asking_for_the_vision_tower_fails_before_anything_is_written(
        self, tmp_path: Path, capsys
    ):
        from local_ocr import cli

        pool_file = tmp_path / "pool.jsonl"
        pool_file.write_text(pool.to_jsonl(pool.Pool(examples=[example()])), encoding="utf-8")
        prompt = tmp_path / "prompt.txt"
        prompt.write_text(PROMPT, encoding="utf-8")
        out = tmp_path / "run"
        code = cli.main(
            [
                "finetune",
                "--pool",
                str(pool_file),
                "--prompt-file",
                str(prompt),
                "--base",
                "reader-a",
                "--revision",
                "b5c3f0a",
                "--out",
                str(out),
                "--vision-layers",
            ]
        )
        assert code == 2
        assert "vLLM" in capsys.readouterr().err
        assert not out.exists()
