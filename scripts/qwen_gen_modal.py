"""Same-seed open-weight renders (Qwen-Image, Apache-2.0) of prompts the pipeline already rewrote, on Modal H100s.

    modal run scripts/qwen_gen_modal.py [--seeds 5491897,1234] [--limit N]

Prompt sets (no new LLM calls; every rewrite comes from an earlier run):
  survey/<idx>  survey-closed + survey-surface: base, naive, simple, surface, v4f
  pilot/<idx>   pilot runs (seed 5491897): base, v1 (pilot-regen attempt 1), v2 (pilot-v2-regen attempt 1),
                v3 (pilot-v3 attempt 1)
Each (prompt, seed) renders once at 1024x1024, 30 steps, true CFG 4.0, empty negative prompt, so the base and
every rewrite of a prompt share the seed. Writes runs/qwen-sameseed/<set>/<idx>/<cond>_<seed>.png and
runs/qwen-sameseed/manifest.json; scripts/vqa_matrix.py picks it up as design "qwen_sameseed".
Weights (~58 GB) are cached in the Modal Volume "adplace-hf-cache".
"""
import glob
import json
from pathlib import Path

import modal

app = modal.App("adplace-qwen-gen")
hf_cache = modal.Volume.from_name("adplace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "diffusers==0.35.1", "transformers==4.56.1", "accelerate==1.10.1",
                 "sentencepiece", "protobuf", "safetensors", "pillow", "huggingface-hub[hf_transfer]==0.34.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
STEPS, CFG, SIZE = 30, 4.0, 1024


@app.cls(image=image, gpu="H100", volumes={"/cache": hf_cache}, timeout=3600, max_containers=4)
class Gen:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
        hf_cache.commit()

    @modal.method()
    def render(self, job: dict) -> dict:
        import io
        import torch
        img = self.pipe(prompt=job["prompt"], negative_prompt=" ", width=SIZE, height=SIZE,
                        num_inference_steps=STEPS, true_cfg_scale=CFG,
                        generator=torch.Generator(device="cuda").manual_seed(job["seed"])).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {**job, "png": buf.getvalue()}


def collect(runs: Path) -> list[dict]:
    sets = []
    for sp in sorted(glob.glob(str(runs / "survey-closed/[0-9][0-9][0-9]/state.json"))):
        s = json.loads(Path(sp).read_text())
        conds = {"base": s["prompt"]}
        for c in ("naive", "v4f"):
            if isinstance(s.get("cond", {}).get(c), dict) and s["cond"][c].get("prompt"):
                conds[c] = s["cond"][c]["prompt"]
        ss = runs / "survey-surface" / Path(sp).parent.name / "state.json"
        if ss.exists():
            t = json.loads(ss.read_text())
            for c in ("simple", "surface"):
                if isinstance(t.get("cond", {}).get(c), dict) and t["cond"][c].get("prompt"):
                    conds[c] = t["cond"][c]["prompt"]
        sets.append({"set": f"survey/{s['idx']}", "brand": s.get("brand"), "conds": conds})
    for rp in sorted(glob.glob(str(runs / "pilot-v3/[0-9][0-9][0-9]/record.json"))):
        r = json.loads(Path(rp).read_text())
        if not r.get("attempts"):
            continue
        conds = {"base": r["prompt"]}
        for c, rd in (("v1", "pilot-regen"), ("v2", "pilot-v2-regen"), ("v3", "pilot-v3")):
            q = runs / rd / Path(rp).parent.name / "record.json"
            if q.exists():
                a = (json.loads(q.read_text()).get("attempts") or [{}])[0]
                mb = a.get("made_by") or {}
                p = (mb.get("prompt") if isinstance(mb, dict) else None) or a.get("prompt")
                if p:
                    conds[c] = p
        sets.append({"set": f"pilot/{Path(rp).parent.name}", "brand": r.get("brand"), "conds": conds})
    return sets


@app.local_entrypoint()
def main(seeds: str = "5491897,1234", limit: int = 0, runs: str = "runs"):
    runs_p = Path(runs)
    out = runs_p / "qwen-sameseed"
    sets = collect(runs_p)[: limit or None]
    jobs = [{"set": s["set"], "cond": c, "prompt": p, "seed": int(sd)}
            for s in sets for c, p in s["conds"].items() for sd in seeds.split(",")]
    todo = [j for j in jobs if not (out / j["set"] / f"{j['cond']}_{j['seed']}.png").exists()]
    print(f"{len(sets)} prompt sets, {len(jobs)} renders ({len(todo)} to do) on Qwen-Image, {STEPS} steps")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"model": "Qwen/Qwen-Image", "steps": STEPS, "cfg": CFG,
                                                   "size": SIZE, "sets": sets, "seeds": seeds}, indent=1))
    for i, r in enumerate(Gen().render.map(todo, order_outputs=False), 1):
        p = out / r["set"] / f"{r['cond']}_{r['seed']}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r["png"])
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} rendered")
