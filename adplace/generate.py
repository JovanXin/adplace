"""Image backends: Comfy Cloud (your FLUX.2 / Qwen workflows), OpenRouter image models, mock."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import requests


class ComfyCloud:
    """Runs an API-format workflow exported from ComfyUI (Workflow > Export (API)).

    The positive prompt node and seed node are auto-detected, or set them in config.
    Docs: https://docs.comfy.org/development/cloud/api-reference
    """

    def __init__(self, api_key: str, base_url: str, workflow_path: str, ledger,
                 prompt_node: str | None = None, seed_node: str | None = None):
        if not api_key:
            raise RuntimeError("COMFY_API_KEY is not set")
        self.key, self.base, self.ledger = api_key, base_url.rstrip("/"), ledger
        self.workflow = json.loads(Path(workflow_path).read_text())
        if "nodes" in self.workflow and "links" in self.workflow:
            raise RuntimeError(f"{workflow_path} is a UI-format workflow. In ComfyUI use "
                               "Workflow > Export (API) and point config at that file.")
        self.prompt_node = prompt_node or self._find_positive()
        self.seed_node = seed_node or self._find_seed()

    # -- workflow introspection -------------------------------------------
    def _find_positive(self) -> str:
        wf = self.workflow

        def follow(ref, depth=0):
            if not isinstance(ref, list) or depth > 6:
                return None
            nid = str(ref[0])
            node = wf.get(nid, {})
            ins = node.get("inputs", {})
            if isinstance(ins.get("text"), str) or isinstance(ins.get("prompt"), str):
                return nid
            for key in ("conditioning", "positive", "clip_conditioning"):
                if key in ins:
                    r = follow(ins[key], depth + 1)
                    if r:
                        return r
            return None

        for nid, node in wf.items():
            ins = node.get("inputs", {})
            for key in ("positive", "conditioning"):
                if key in ins and node.get("class_type") not in ("CLIPTextEncode",):
                    r = follow(ins[key])
                    if r:
                        return r
        for nid, node in wf.items():
            if "TextEncode" in node.get("class_type", "") and isinstance(node["inputs"].get("text"), str):
                return nid
        raise RuntimeError("Couldn't find the positive prompt node; set generation.comfy_prompt_node")

    def _find_seed(self) -> str | None:
        for nid, node in self.workflow.items():
            ins = node.get("inputs", {})
            if any(k in ins and not isinstance(ins[k], list) for k in ("seed", "noise_seed")):
                return nid
        return None

    def _text_key(self, nid):
        ins = self.workflow[nid]["inputs"]
        return "text" if "text" in ins else "prompt"

    def build(self, prompt: str, seed: int, width: int | None = None, height: int | None = None,
              input_image_name: str | None = None) -> dict:
        wf = copy.deepcopy(self.workflow)
        wf[self.prompt_node]["inputs"][self._text_key(self.prompt_node)] = prompt
        if self.seed_node:
            ins = wf[self.seed_node]["inputs"]
            ins["noise_seed" if "noise_seed" in ins else "seed"] = int(seed)
        for node in wf.values():
            ins = node.get("inputs", {})
            if width and "width" in ins and not isinstance(ins["width"], list):
                ins["width"] = width
            if height and "height" in ins and not isinstance(ins["height"], list):
                ins["height"] = height
            if input_image_name and node.get("class_type") == "LoadImage":
                ins["image"] = input_image_name
        return wf

    # -- HTTP -------------------------------------------------------------
    @property
    def _h(self):
        return {"X-API-Key": self.key}

    def upload(self, img: bytes, name: str) -> str:
        r = requests.post(f"{self.base}/api/upload/image", headers=self._h,
                          files={"image": (name, img, "image/png")},
                          data={"type": "input", "overwrite": "true"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    def run(self, wf: dict, timeout: float = 3600) -> bytes:
        # The timeout includes time waiting in the account's queue (1 concurrent job on Standard).
        self.ledger.check(needs_image=True)
        r = requests.post(f"{self.base}/api/prompt", headers={**self._h, "Content-Type": "application/json"},
                          data=json.dumps({"prompt": wf}), timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Comfy submit failed {r.status_code}: {r.text[:500]}")
        pid = r.json()["prompt_id"]
        t0 = time.time()
        while True:
            try:
                s = requests.get(f"{self.base}/api/job/{pid}/status", headers=self._h, timeout=30)
                status = s.json().get("status") if s.ok else None
            except (requests.RequestException, ValueError):
                status = None          # transient network error: keep polling until the timeout
            if status == "success":
                break
            if status in ("error", "non_retryable_error", "lost", "cancelled"):
                raise RuntimeError(f"Comfy job {pid} ended with {status}: {s.text[:500]}")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Comfy job {pid} timed out")
            time.sleep(3)
        outputs = self._outputs(pid)
        for node_out in outputs.values():
            for im in node_out.get("images", []):
                if im.get("type", "output") == "output":
                    self.ledger.record("comfy_generate", "comfy-cloud", images=1, prompt_id=pid)
                    return self._download(im["filename"], im.get("subfolder", ""))
        raise RuntimeError(f"Comfy job {pid} finished but no output image found: {json.dumps(outputs)[:400]}")

    def _outputs(self, pid: str) -> dict:
        for path in (f"/api/history_v2/{pid}", f"/api/history/{pid}", f"/api/jobs/{pid}"):
            r = requests.get(f"{self.base}{path}", headers=self._h, timeout=30)
            if not r.ok:
                continue
            j = r.json()
            node = j.get(pid, j)
            outs = node.get("outputs")
            if outs:
                return outs
        raise RuntimeError(f"Couldn't read outputs for {pid} (tried history_v2/history/jobs)")

    def _download(self, filename: str, subfolder: str) -> bytes:
        r = requests.get(f"{self.base}/api/view", headers=self._h, allow_redirects=False, timeout=60,
                         params={"filename": filename, "subfolder": subfolder, "type": "output"})
        if r.status_code in (301, 302, 303, 307, 308):
            f = requests.get(r.headers["Location"], timeout=120)   # signed URL: no auth header
            f.raise_for_status()
            return f.content
        r.raise_for_status()
        return r.content

    # -- public -----------------------------------------------------------
    def generate(self, prompt: str, seed: int, width: int, height: int) -> bytes:
        return self.run(self.build(prompt, seed, width, height))


class ComfyEdit(ComfyCloud):
    """Edit workflow: must contain a LoadImage node plus a text prompt node."""

    def edit(self, image: bytes, instruction: str, seed: int) -> bytes:
        name = self.upload(image, f"adplace_{int(time.time() * 1000)}.png")
        return self.run(self.build(instruction, seed, input_image_name=name))


class OpenRouterImage:
    def __init__(self, llm, model: str):
        self.llm, self.model = llm, model

    def generate(self, prompt: str, seed: int, width: int, height: int) -> bytes:
        return self.llm.image(self.model, prompt, task="generate", width=width, height=height, seed=seed)

    def edit(self, image: bytes, instruction: str, seed: int) -> bytes:
        return self.llm.image(self.model, instruction, task="edit", input_image=image)
