#!/usr/bin/env python
import pathlib, os

RUN_DIR = pathlib.Path(
    "/scratch/network/pw5115/my_less_project/implicit-ins-improved/out/llama2-7b-p0.05-lora-seed3"
)

for idx_path in RUN_DIR.rglob("optimizer.pt.idx"):
    pt_path = idx_path.with_suffix("")  # strip ".idx"
    if pt_path.exists():                       # keep the named‑key file just in case
        pt_path.rename(pt_path.with_suffix(".named"))
    idx_path.rename(pt_path)
    print(f"✓ restored {pt_path.relative_to(RUN_DIR.parent)}")

