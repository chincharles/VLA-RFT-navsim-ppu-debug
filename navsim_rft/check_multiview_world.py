"""Render four-camera world-model diagnostics without changing model weights.

Each contact sheet contains, for every camera, the real future frame, the
teacher-forced reconstruction and the free autoregressive rollout.  The
rollout receives one action block and is never re-planned per frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .checkpoint import load
from .data import MultiViewSceneDataset
from .rewards import ImageReward
from .world import TokenWorld


CAMERAS = ("front", "rear", "left", "right")


def _pil(x):
    x = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def _sheet(truth, teacher, free, token, out):
    # rows=cameras, columns=truth/teacher/free; preserve all 8 future frames
    # in a compact contact sheet so blur and temporal drift are visible.
    t = truth.shape[1]
    cell_w, cell_h = truth.shape[-1], truth.shape[-2]
    sheet = Image.new("RGB", (3 * t * cell_w, 4 * (cell_h + 20)), "black")
    draw = ImageDraw.Draw(sheet)
    labels = ("truth", "teacher", "free")
    for c, name in enumerate(CAMERAS):
        draw.text((0, c * (cell_h + 20)), name, fill="white")
        for col, seq in enumerate((truth[c], teacher[c], free[c])):
            for step in range(t):
                x = (col * t + step) * cell_w
                y = c * (cell_h + 20) + 20
                sheet.paste(_pil(seq[step]), (x, y))
                if c == 0:
                    draw.text((x + 3, 2), f"{labels[col]}-{step+1}", fill="white")
    sheet.save(out / f"{token}.png")


def main():
    p = argparse.ArgumentParser()
    for key in ("checkpoint", "tokenizer", "manifest", "stats", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--limit", type=int, default=4)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    ds = MultiViewSceneDataset(a.manifest, a.stats)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("metadata", {}).get("views") != list(ds.EXPECTED_VIEWS):
        raise ValueError("Checkpoint is not a four-camera world-model checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = TokenWorld(a.tokenizer, ds.stats).to(device)
    load(a.checkpoint, {"world": wm})
    wm.eval().requires_grad_(False)
    reward = ImageReward().to(device)
    rows = []
    with torch.inference_mode():
        for i in range(min(len(ds), a.limit)):
            item = ds[i]
            views = item["views"][:, -9:].to(device)
            deltas = torch.tensor(item["deltas"], dtype=torch.float32, device=device)[None]
            valid = item["valid"].to(device)[None]
            truth_all, teacher_all, free_all = [], [], []
            camera_rows = []
            for cam in range(4):
                truth = views[cam, 1:][None]
                initial = views[cam, :1]
                teacher = wm.rollout(initial, deltas, teacher_video=views[cam : cam + 1], camera_id=cam)
                free = wm.rollout(initial, deltas, camera_id=cam)
                _, tm = reward(teacher, truth, valid)
                _, fm = reward(free, truth, valid)
                truth_all.append(truth[0])
                teacher_all.append(teacher[0])
                free_all.append(free[0])
                camera_rows.append(dict(camera=ds.EXPECTED_VIEWS[cam],
                    teacher_l1=float(tm["l1"].mean()), teacher_lpips=float(tm["lpips"].mean()),
                    free_l1=float(fm["l1"].mean()), free_lpips=float(fm["lpips"].mean())))
            truth = torch.stack(truth_all)
            teacher = torch.stack(teacher_all)
            free = torch.stack(free_all)
            _sheet(truth, teacher, free, item["token"], out)
            rows.append(dict(token=item["token"], cameras=camera_rows))
    (out / "quality.json").write_text(json.dumps(dict(views=list(ds.EXPECTED_VIEWS), rows=rows), indent=2))
    print(json.dumps(dict(output=str(out), scenes=len(rows), views=list(ds.EXPECTED_VIEWS)), indent=2))


if __name__ == "__main__":
    main()
