"""Recover manifest/statistics after image export completed before JSON writing."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from . import NAVSIM_COMMIT
from .data import build_loader
from .export_multiview import CAMERAS, jsonable
from .geometry import ActionStats


def main():
    p = argparse.ArgumentParser()
    for k in ("root", "logs", "sensors", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--limit", type=int, default=800)
    a = p.parse_args()
    out = Path(a.out)
    files = sorted(out.glob("*.npz"))
    if not files:
        raise ValueError(f"No exported NPZ files found in {out}")
    loader = build_loader(a.root, a.logs, a.sensors, "navtrain", None, views=CAMERAS)
    records, trajectories = [], []
    for index, path in enumerate(files[:a.limit], 1):
        token = path.stem
        if token not in loader.tokens:
            raise ValueError(f"Exported token is absent from NAVSIM loader: {token}")
        with np.load(path, allow_pickle=False) as z:
            views = z["views"]
            timestamps = z["timestamps"].tolist()
            if views.ndim != 5 or views.shape[0] != 4 or views.shape[2:] != (256, 256, 3):
                raise ValueError(f"{path}: unexpected views shape {views.shape}")
        raw = loader.scene_frames_dicts[token]
        h = 4
        log_name = raw[h - 1]["log_name"]
        command = np.asarray(raw[h - 1]["driving_command"], dtype=np.float32).reshape(-1)
        if command.shape != (4,):
            raise ValueError(f"{token}: unexpected navigation command shape {command.shape}")
        text = "Drive according to the official navigation command vector: " + ",".join(str(int(v)) for v in command)
        if int(hashlib.sha256(log_name.encode()).hexdigest()[:8], 16) % 10 == 0:
            raise ValueError(f"{token} belongs to held-out validation log; refusing train stats")
        # Avoid Scene.from_scene_dict_list here: it constructs a map API for
        # every scene. The exported NPZ already contains all image data, and
        # the raw log stores the ego poses needed for normalization statistics.
        from pyquaternion import Quaternion
        origin = raw[h - 1]
        oq = Quaternion(*origin["ego2global_rotation"])
        ox, oy, oyaw = origin["ego2global_translation"][0], origin["ego2global_translation"][1], oq.yaw_pitch_roll[0]
        c, s = np.cos(oyaw), np.sin(oyaw)
        poses = []
        for frame in raw[h : h + 8]:
            q = Quaternion(*frame["ego2global_rotation"])
            dx = float(frame["ego2global_translation"][0]) - float(ox)
            dy = float(frame["ego2global_translation"][1]) - float(oy)
            yaw = float(q.yaw_pitch_roll[0])
            poses.append([c * dx + s * dy, -s * dx + c * dy, (yaw - oyaw + np.pi) % (2 * np.pi) - np.pi])
        poses = np.asarray(poses, dtype=np.float64)
        camera_timestamps = {c: [jsonable(raw[i]["cams"].get(c.upper())) for i in range(views.shape[1])] for c in CAMERAS}
        records.append(dict(token=token, scene_token=raw[h - 1]["scene_token"],
            log_name=log_name, text=text, file=path.name, views=list(CAMERAS),
            view_order="front,rear,left,right", timestamps=timestamps,
            camera_timestamps=camera_timestamps))
        trajectories.append(poses)
        if index == 1 or index % 100 == 0:
            print(f"recovered={index}/{min(len(files), a.limit)}", flush=True)
    if len(records) < a.limit:
        print(f"warning: recovered {len(records)} files (requested {a.limit})")
    manifest = dict(navsim_commit=NAVSIM_COMMIT, split="navtrain", role="train", dt=.5,
        horizon=4., views=list(CAMERAS), view_order="front,rear,left,right",
        shape="[camera,time,height,width,channel]", records=records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ActionStats.fit(trajectories, [r["token"] for r in records], "train").save(out / "stats.json")
    print(json.dumps(dict(recovered=len(records), output=str(out)), indent=2))


if __name__ == "__main__":
    main()
