"""The default reader: one image and one prompt against a local server.

vLLM and SGLang both speak the OpenAI vision API, so this one adapter covers
both and a second model is a URL change rather than a rewrite.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from local_ocr.batch import Refused

# Phrases that mean the model declined rather than failed. Kept short and
# specific: a page of mathematics can legitimately contain almost any sentence,
# so a loose match here would refuse pages that were read correctly.
REFUSALS = (
    "i'm sorry, i can't",
    "i cannot assist",
    "i can't help with that",
    "unable to process this image",
)


def why(response: httpx.Response) -> str:
    """What the server said, rather than what its status code was.

    httpx raises "Client error '400 Bad Request' for url ...", which is the same
    sentence whatever went wrong, and a batch log full of it says only that two
    hundred pages failed. The server's own message is the useful part: the one
    that produced this function read "Failed to load image: cannot identify
    image file", which is a sentence somebody can act on.
    """
    text = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            text = str(error["message"])
        elif isinstance(error, str):
            text = error
        elif payload.get("message"):
            text = str(payload["message"])
    return f"HTTP {response.status_code}: {text[:400]}" if text else f"HTTP {response.status_code}"


def _data_url(image: Path) -> str:
    kind, _ = mimetypes.guess_type(image.name)
    payload = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{kind or 'image/png'};base64,{payload}"


@dataclass
class OpenAIVisionReader:
    model: str = "reader"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "not-needed"
    # Transcription, not composition. There is nothing to be creative about, and
    # a nonzero temperature turns an already hard formula into a lottery.
    temperature: float = 0.0
    max_tokens: int = 8192
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No timeout here. The per page bound is `--timeout` and it is
            # enforced by the batch, so that one slow page is refused by the
            # same code path as one that fails.
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)
        return self._client

    async def read(self, image: Path, prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # One page per request, always. The queue is per page,
                        # the acceptance rules are per page and the page file is
                        # per page, so batching two into one prompt would let a
                        # single bad answer poison both.
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = await self.client().post(
            "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.is_error:
            raise Refused(why(response))
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise Refused("the server returned no choices")
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            raise Refused("the server returned an empty message")
        lowered = text.lower()
        if any(phrase in lowered for phrase in REFUSALS):
            raise Refused(text.splitlines()[0][:120])
        if choices[0].get("finish_reason") == "length":
            # A truncated page fails RuleShort on the Go side anyway, and it is
            # much cheaper to say so here than to ship it and read it twice.
            raise Refused("the answer hit the token limit and is truncated")
        return text
