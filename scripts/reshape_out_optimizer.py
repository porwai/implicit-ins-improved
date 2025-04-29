#!/usr/bin/env python
"""
Rewrite optimizer.pt in each checkpoint so that state['state'] is keyed by
parameter *names* instead of integer indices.  The original file is preserved
as optimizer.pt.idx.
"""
import os, torch, pathlib, textwrap
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------------------
BASE_MODEL  = "/scratch/network/pw5115/my_less_project/Llama-2-7b-hf"                 # base HF model
RUN_DIR     = "/scratch/network/pw5115/my_less_project/implicit-ins-improved/out/llama2-7b-p0.05-lora-seed3"
CHECKPOINTS = [2468, 4936, 7405, 9872]                  # warm‑up checkpoints
# ---------------------------------------------------------------------------

def rekey_optimizer(ckpt_dir: str) -> None:
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    bak_path = os.path.join(ckpt_dir, "optimizer.pt.idx")  # backup name

    if not os.path.isfile(opt_path):
        print(f"⚠  {opt_path} not found; skipping.")
        return

    # 1. rename original -> backup
    os.rename(opt_path, bak_path)

    # 2. load raw state (still idx‑keyed)
    opt_sd = torch.load(bak_path, map_location="cpu")
    state_idx = opt_sd["state"]

    # 3. load model+adapter so we know the param order
    base   = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model  = PeftModel.from_pretrained(base, ckpt_dir)

    idx2name = {i: n for i, (n, _) in enumerate(model.named_parameters())}

    # 4. re‑key
    state_name = {idx2name[i]: v for i, v in state_idx.items() if i in idx2name}
    opt_sd["state"] = state_name

    # 5. save new optimizer.pt
    torch.save(opt_sd, opt_path)
    print(f"✓ {pathlib.Path(ckpt_dir).name}: optimizer.pt rewritten "
          f"(backup → {bak_path})")

if __name__ == "__main__":
    for step in CHECKPOINTS:
        ckpt = os.path.join(RUN_DIR, f"checkpoint-{step}")
        rekey_optimizer(ckpt)
