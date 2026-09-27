"""OpenRouter client: JSON chat, vision, image generation/editing, spend ledger.

Every call asks OpenRouter to return its real cost (usage.include) and the cost is
appended to a ledger. A call is refused once the ledger reaches the configured cap.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

OR_BASE = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Spend is summed over the whole ledger file (the cap is global across runs);
    the image count only covers records tagged with this run_id (a per-run cap)."""
    path: Path
    cap_usd: float
    max_images: int
    run_id: str = ""
    spent: float = 0.0
    images: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def begin(self, tag: str = ""):
        """Start per-thread accounting (one prompt = one worker thread)."""
        self._local.acc, self._local.tag = 0.0, tag

    def thread_spent(self) -> float:
        return getattr(self._local, "acc", 0.0)

    def __post_init__(self):
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.spent += float(rec.get("cost", 0) or 0)
                if rec.get("run_id", "") == self.run_id:
                    self.images += int(rec.get("images", 0) or 0)

    def check(self, needs_image: bool = False):
        if self.spent >= self.cap_usd:
            raise BudgetExceeded(f"OpenRouter spend ${self.spent:.3f} reached cap ${self.cap_usd:.2f}")
        if needs_image and self.images >= self.max_images:
            raise BudgetExceeded(f"Image generation cap reached ({self.images}/{self.max_images})")

    def record(self, task: str, model: str, cost: float = 0.0, images: int = 0, **extra):
        with self._lock:
            self.spent += cost
            self.images += images
            self._local.acc = getattr(self._local, "acc", 0.0) + cost
            if getattr(self._local, "tag", ""):
                extra.setdefault("tag", self._local.tag)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"t": time.time(), "run_id": self.run_id, "task": task,
                                    "model": model, "cost": cost, "images": images, **extra}) + "\n")


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ``` fences and prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"No JSON object in model reply: {text[:300]}")


def image_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"


def data_url_to_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


class OpenRouter:
    def __init__(self, api_key: str, ledger: Ledger, app_name: str = "adplace-thesis"):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.key = api_key
        self.ledger = ledger
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                        "X-Title": app_name}
        self._models: list[dict] | None = None

    # ---- model discovery -------------------------------------------------
    def models(self) -> list[dict]:
        if self._models is None:
            r = requests.get(f"{OR_BASE}/models", timeout=60)
            r.raise_for_status()
            self._models = r.json()["data"]
        return self._models

    def resolve(self, wanted: str | list[str], need_image_in=False, need_image_out=False) -> str:
        wanted = [wanted] if isinstance(wanted, str) else list(wanted)
        ms = self.models()

        def ok(m):
            arch = m.get("architecture", {})
            if need_image_in and "image" not in arch.get("input_modalities", []):
                return False
            if need_image_out and "image" not in arch.get("output_modalities", []):
                return False
            return True

        ids = {m["id"]: m for m in ms if ok(m)}
        for w in wanted:
            if w in ids:
                return w
        for w in wanted:
            cands = sorted([i for i in ids if w.split("/")[-1] in i], key=len)
            if cands:
                return cands[0]
        raise RuntimeError(f"None of {wanted} found on OpenRouter "
                           f"(image_in={need_image_in}, image_out={need_image_out}). "
                           f"Run `python -m adplace models` to see options.")

    def key_status(self) -> dict:
        r = requests.get(f"{OR_BASE}/key", headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", {})

    # ---- core call -------------------------------------------------------
    def _post(self, body: dict, task: str, images_expected: int = 0, retries: int = 3) -> dict:
        self.ledger.check(needs_image=images_expected > 0)
        body = {**body, "usage": {"include": True}}
        last = None
        for attempt in range(retries):
            try:
                r = requests.post(f"{OR_BASE}/chat/completions", headers=self.headers,
                                  data=json.dumps(body), timeout=300)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"{r.status_code}: {r.text[:300]}")
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise RuntimeError(str(data["error"])[:500])
                usage = data.get("usage") or {}
                cost = float(usage.get("cost") or 0.0)
                self.ledger.record(task, body.get("model", "?"), cost=cost, images=images_expected,
                                   prompt_tokens=usage.get("prompt_tokens"),
                                   completion_tokens=usage.get("completion_tokens"))
                return data
            except requests.RequestException as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenRouter call failed for {task}: {last}")

    def chat_json(self, model: str, system: str, user: str | list, task: str,
                  temperature: float = 0.2, images: list[bytes] | None = None) -> dict:
        content: list | str
        if images:
            content = [{"type": "text", "text": user if isinstance(user, str) else json.dumps(user)}]
            for im in images:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(im)}})
        else:
            content = user if isinstance(user, str) else json.dumps(user, indent=1)
        body = {"model": model, "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}]}
        data = self._post(body, task)
        text = data["choices"][0]["message"].get("content") or ""
        try:
            return extract_json(text)
        except (ValueError, json.JSONDecodeError):
            # one repair attempt, without response_format (some providers ignore it)
            body.pop("response_format", None)
            body["messages"].append({"role": "assistant", "content": text})
            body["messages"].append({"role": "user", "content": "Return ONLY the JSON object, nothing else."})
            data = self._post(body, task + ":json_repair")
            return extract_json(data["choices"][0]["message"].get("content") or "")

    def image(self, model: str, prompt: str, task: str, input_image: bytes | None = None,
              width: int = 1024, height: int = 1024) -> bytes:
        """Text->image, or image+instruction->image (edit) via an image-output model."""
        if input_image is not None:
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": image_to_data_url(input_image)}}]
        else:
            content = prompt
        body = {"model": model, "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": content}]}
        if width != height:
            body["image_config"] = {"aspect_ratio": _aspect(width, height)}
        data = self._post(body, task, images_expected=1)
        msg = data["choices"][0]["message"]
        imgs = msg.get("images") or []
        if not imgs:
            raise RuntimeError(f"{model} returned no image. Text: {(msg.get('content') or '')[:300]}")
        return data_url_to_bytes(imgs[0]["image_url"]["url"])


def _aspect(w: int, h: int) -> str:
    from math import gcd
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def load_env(dotenv: Path):
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def png_bytes(pil_img) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
