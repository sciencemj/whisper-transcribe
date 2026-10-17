"""Summarize a meeting transcript with a locally running LLM.

Talks to Ollama (`/api/chat`) or any OpenAI-compatible server
(LM Studio, llama.cpp `llama-server`, vLLM, ...) over plain HTTP using
only the standard library. By default only loopback addresses are
allowed so meeting content never leaves the machine.
"""

import ipaddress
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse

LogCb = Callable[[str], None]

DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:1234/v1",
}
# LG AI Research's EXAONE is trained heavily on Korean and runs well on a laptop.
DEFAULT_MODEL = "exaone3.5:7.8b"

SYSTEM_PROMPT = """당신은 회의록 작성 전문가입니다.
입력은 음성 인식(STT)으로 자동 생성된 한국어 회의 녹취록이므로 오탈자나 잘못 인식된 단어가 있을 수 있습니다.
문맥상 명백한 인식 오류는 바로잡되, 녹취록에 없는 내용은 절대 지어내지 마세요.
담당자나 기한이 언급되지 않았다면 '미정'으로 표기하세요.
모든 출력은 한국어 마크다운으로 작성하세요."""

CHUNK_PROMPT = """다음은 회의 녹취록의 일부({index}/{total})입니다.
이 부분에서 나온 안건, 핵심 논의 내용, 결정 사항, 할 일(담당자/기한 포함), 미해결 질문을
빠짐없이 간결한 글머리표로 정리하세요. 시간 표시([HH:MM:SS])는 중요한 항목에 함께 남겨 두세요.

<녹취록>
{text}
</녹취록>"""

FINAL_PROMPT = """아래 {source_desc}를 바탕으로 최종 회의록을 작성하세요.
반드시 다음 형식과 제목을 그대로 사용하세요.

## 회의 요약
(회의 목적과 결과를 2~3문장으로)

## 주요 안건
1. ...

## 논의 내용
### (안건 1 제목)
- ...

## 결정 사항
- ...

## 액션 아이템
| 담당자 | 할 일 | 기한 |
|---|---|---|
| ... | ... | ... |

## 추후 논의 사항
- ...

해당 내용이 없는 섹션은 '- 없음'으로 표기하세요.

<{tag}>
{text}
</{tag}>"""


class LLMError(RuntimeError):
    pass


def _is_local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class LocalLLM:
    model: str = DEFAULT_MODEL
    backend: str = "ollama"
    url: Optional[str] = None
    num_ctx: int = 8192
    temperature: float = 0.2
    timeout: float = 900.0
    allow_remote: bool = False

    def __post_init__(self):
        if self.backend not in DEFAULT_URLS:
            raise ValueError(f"unknown LLM backend: {self.backend}")
        self.url = (self.url or DEFAULT_URLS[self.backend]).rstrip("/")
        if not self.allow_remote and not _is_local(self.url):
            raise LLMError(
                f"{self.url} is not a local address; meeting content would leave this machine. "
                "Pass --allow-remote-llm if this is intended."
            )

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise LLMError(f"LLM server returned {e.code}: {body}") from e
        except urllib.error.URLError as e:
            hint = (
                "Is Ollama running? Start it with `ollama serve` and pull the model with "
                f"`ollama pull {self.model}`."
                if self.backend == "ollama"
                else "Is the local OpenAI-compatible server running?"
            )
            raise LLMError(f"cannot reach {self.url}: {e.reason}. {hint}") from e

    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.backend == "ollama":
            data = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
                },
            )
            content = data["message"]["content"]
        else:
            data = self._post(
                "/chat/completions",
                {"model": self.model, "messages": messages, "temperature": self.temperature},
            )
            content = data["choices"][0]["message"]["content"]
        return _strip_thinking(content)

    def _get(self, path: str, timeout: float = 5) -> dict:
        with urllib.request.urlopen(urllib.request.Request(self.url + path), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> list[str]:
        """Names of models that run on this machine (best effort).

        Ollama cloud models (which forward prompts to ollama.com) are
        left out unless remote use is allowed.
        """
        try:
            if self.backend == "ollama":
                return [
                    m["name"]
                    for m in self._get("/api/tags")["models"]
                    if self.allow_remote or not m.get("remote_host")
                ]
            return [m["id"] for m in self._get("/models")["data"]]
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            return []

    def ensure_local_model(self) -> None:
        """Fail early unless the model is installed and runs locally.

        Ollama can proxy "cloud" models (e.g. `gpt-oss:20b-cloud`) to
        ollama.com through the same local API, so a localhost URL alone
        does not mean the transcript stays on this machine.
        """
        if self.backend != "ollama" or self.allow_remote:
            return
        try:
            models = self._get("/api/tags")["models"]
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach {self.url}: {e.reason}. Is Ollama running? Start it with `ollama serve`."
            ) from e
        except (OSError, KeyError, ValueError) as e:
            raise LLMError(f"unexpected response from {self.url}/api/tags: {e}") from e

        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        match = next((m for m in models if m.get("name") in (self.model, wanted)), None)
        if match is None:
            raise LLMError(f"model '{self.model}' is not installed. Run `ollama pull {self.model}` first.")
        if match.get("remote_host"):
            raise LLMError(
                f"'{self.model}' is an Ollama cloud model served by {match['remote_host']}; "
                "the transcript would leave this machine. Use a local model instead."
            )


def _strip_thinking(text: str) -> str:
    # Reasoning models (qwen3, deepseek-r1, ...) emit their chain of thought first.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def split_chunks(transcript: str, max_chars: int) -> list[str]:
    """Split on line boundaries into chunks of at most ~max_chars."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in transcript.splitlines():
        if cur and size + len(line) > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def summarize(
    llm: LocalLLM,
    transcript: str,
    chunk_chars: int = 6000,
    log: LogCb = print,
) -> str:
    """Map-reduce summary: notes per chunk, then one structured minutes doc."""
    llm.ensure_local_model()
    chunks = split_chunks(transcript, chunk_chars)
    if len(chunks) <= 1:
        log(f"summarizing with {llm.model}...")
        return llm.chat(
            SYSTEM_PROMPT,
            FINAL_PROMPT.format(source_desc="회의 녹취록", tag="녹취록", text=transcript),
        )

    notes = []
    for i, chunk in enumerate(chunks, 1):
        log(f"summarizing part {i}/{len(chunks)} with {llm.model}...")
        notes.append(
            llm.chat(SYSTEM_PROMPT, CHUNK_PROMPT.format(index=i, total=len(chunks), text=chunk))
        )

    log("writing final minutes...")
    merged = "\n\n".join(f"### 파트 {i}\n{n}" for i, n in enumerate(notes, 1))
    return llm.chat(
        SYSTEM_PROMPT,
        FINAL_PROMPT.format(source_desc="회의 파트별 정리 노트", tag="정리노트", text=merged),
    )
