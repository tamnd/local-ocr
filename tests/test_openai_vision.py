"""The reader that talks to a local server, with the server stubbed.

httpx has a mock transport, so every one of these is the real request building
and the real response handling against a server that answers however the test
needs it to.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from local_ocr.backends.openai_vision import OpenAIVisionReader, why
from local_ocr.batch import Refused

PAGE = b"\x89PNG\r\n\x1a\nnot really a png, and the server never sees it as one"


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "0042.png"
    path.write_bytes(PAGE)
    return path


def reader_answering(handler, **kwargs) -> OpenAIVisionReader:
    reader = OpenAIVisionReader(model="reader-a", **kwargs)
    reader._client = httpx.AsyncClient(
        base_url=reader.base_url, transport=httpx.MockTransport(handler)
    )
    return reader


def answer(text: str, finish: str = "stop", usage: dict | None = None) -> httpx.Response:
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


class TestWhy:
    def test_the_servers_own_message_is_what_comes_out(self) -> None:
        # This is the shape vLLM sends, and this is the sentence that was
        # invisible for a run of two hundred refused pages.
        response = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Failed to load image: cannot identify image file",
                    "type": "BadRequestError",
                    "code": 400,
                }
            },
        )
        said = why(response)
        assert "cannot identify image file" in said
        assert "400" in said

    def test_an_error_that_is_a_string_is_read_too(self) -> None:
        assert "went wrong" in why(httpx.Response(500, json={"error": "something went wrong"}))

    def test_a_body_that_is_not_json_still_reaches_the_log(self) -> None:
        assert "gateway" in why(httpx.Response(502, text="bad gateway"))

    def test_no_body_at_all_leaves_the_status(self) -> None:
        assert why(httpx.Response(503)) == "HTTP 503"

    def test_a_long_body_is_cut_rather_than_filling_the_log(self) -> None:
        # One line a page in a log somebody reads by tailing it.
        assert len(why(httpx.Response(400, text="x" * 5000))) < 500


class TestRead:
    def test_a_page_is_sent_as_one_image_and_one_prompt(self, image: Path) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return answer("PROPOSITION 7.")

        reader = reader_answering(handler)
        assert asyncio.run(reader.read(image, "Read this page.")) == "PROPOSITION 7."
        content = seen["messages"][0]["content"]  # type: ignore[index]
        assert [part["type"] for part in content] == ["image_url", "text"]
        assert content[1]["text"] == "Read this page."
        assert seen["temperature"] == 0.0

    def test_the_image_arrives_whole(self, image: Path) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return answer("read")

        asyncio.run(reader_answering(handler).read(image, "Read this page."))
        url = seen["messages"][0]["content"][0]["image_url"]["url"]  # type: ignore[index]
        head, _, payload = url.partition(",")
        assert head == "data:image/png;base64"
        assert base64.b64decode(payload) == PAGE

    def test_no_answer_length_is_asked_for_unless_one_is_set(self, image: Path) -> None:
        # Asking for a fixed number of output tokens is asking the server to
        # reject the page whenever that number is most of its window, which is
        # how DeepSeek-OCR, whose window is 8192, refused two hundred pages.
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return answer("read")

        asyncio.run(reader_answering(handler).read(image, "Read this page."))
        assert "max_tokens" not in seen

    def test_a_bound_is_sent_when_one_is_asked_for(self, image: Path) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return answer("read")

        asyncio.run(reader_answering(handler, max_tokens=4096).read(image, "Read this page."))
        assert seen["max_tokens"] == 4096

    def test_an_error_is_a_refusal_that_says_what_the_server_said(self, image: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, json={"error": {"message": "Failed to load image: cannot identify image file"}}
            )

        reader = reader_answering(handler)
        with pytest.raises(Refused) as caught:
            asyncio.run(reader.read(image, "Read this page."))
        assert "cannot identify image file" in str(caught.value)

    def test_a_truncated_answer_is_refused_here_rather_than_shipped(self, image: Path) -> None:
        reader = reader_answering(lambda request: answer("half a page", finish="length"))
        with pytest.raises(Refused):
            asyncio.run(reader.read(image, "Read this page."))

    def test_an_empty_answer_is_refused(self, image: Path) -> None:
        reader = reader_answering(lambda request: answer("   "))
        with pytest.raises(Refused):
            asyncio.run(reader.read(image, "Read this page."))

    def test_a_model_that_declines_is_refused_in_its_own_words(self, image: Path) -> None:
        reader = reader_answering(
            lambda request: answer("I'm sorry, I can't help with this image.")
        )
        with pytest.raises(Refused) as caught:
            asyncio.run(reader.read(image, "Read this page."))
        assert "sorry" in str(caught.value).lower()


class TestUsage:
    """What the server charged, which the sidecar carried as zero for a whole run.

    The M6 run wrote 122 sidecars and every one of them said prompt_tokens 0 and
    completion_tokens 0, because the counts were in the response and `read`
    returns a string. They are collected on the side now, and the thing that
    matters most here is that a page nobody counted is told apart from a page
    that cost nothing.
    """

    def test_the_counts_the_server_sent_come_back_once(self, image: Path) -> None:
        reader = reader_answering(
            lambda request: answer("read", usage={"prompt_tokens": 1512, "completion_tokens": 803})
        )
        asyncio.run(reader.read(image, "Read this page."))
        assert reader.usage(image) == (1512, 803)
        # Once, because a second page read into the same sidecar would otherwise
        # inherit the counts of the page before it.
        assert reader.usage(image) is None

    def test_a_server_that_does_not_count_says_nothing_rather_than_zero(self, image: Path) -> None:
        reader = reader_answering(lambda request: answer("read"))
        asyncio.run(reader.read(image, "Read this page."))
        assert reader.usage(image) is None

    def test_a_usage_block_that_is_not_numbers_is_ignored(self, image: Path) -> None:
        reader = reader_answering(lambda request: answer("read", usage={"prompt_tokens": "1512"}))
        asyncio.run(reader.read(image, "Read this page."))
        assert reader.usage(image) is None

    def test_a_refused_page_files_nothing(self, image: Path) -> None:
        reader = reader_answering(
            lambda request: answer(
                "half a page", finish="length", usage={"prompt_tokens": 9, "completion_tokens": 9}
            )
        )
        with pytest.raises(Refused):
            asyncio.run(reader.read(image, "Read this page."))
        assert reader.usage(image) is None

    def test_two_pages_are_counted_apart(self, image: Path, tmp_path: Path) -> None:
        other = tmp_path / "0043.png"
        other.write_bytes(PAGE)
        counts = iter([(100, 10), (200, 20)])

        def handler(request: httpx.Request) -> httpx.Response:
            prompt, completion = next(counts)
            return answer("read", usage={"prompt_tokens": prompt, "completion_tokens": completion})

        reader = reader_answering(handler)
        asyncio.run(reader.read(image, "Read this page."))
        asyncio.run(reader.read(other, "Read this page."))
        assert reader.usage(other) == (200, 20)
        assert reader.usage(image) == (100, 10)

    def test_a_caller_that_never_collects_does_not_grow_without_bound(self, tmp_path: Path) -> None:
        reader = reader_answering(
            lambda request: answer("read", usage={"prompt_tokens": 1, "completion_tokens": 1})
        )
        for n in range(600):
            page = tmp_path / f"{n:04d}.png"
            page.write_bytes(PAGE)
            asyncio.run(reader.read(page, "Read this page."))
        assert len(reader._usage) <= 512
