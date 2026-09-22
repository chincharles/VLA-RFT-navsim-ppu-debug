"""Export four-camera NAVSIM clips for the multiview tokenizer/world-model branch.

The exported tensor layout is [camera, time, channel, height, width]. Cameras are
front, rear, left and right (cam_f0, cam_b0, cam_l0, cam_r0). This exporter does
not change the existing single-view trainer; it provides the audited input format
for the multiview model.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from . import NAVSIM_COMMIT
from .data import build_loader, legal_input
from .geometry import ActionStats

CAMERAS = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")

def main():
    p=argparse.ArgumentParser()
    for k in ("root","logs","sensors","split","role","out"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--limit",type=int,required=True)
    p.add_argument("--exclude")
    a=p.parse_args()
    if a.role not in ("train","val") or a.split != "navtrain":
        raise ValueError("Multiview training export currently supports navtrain train/val only")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    loader=build_loader(a.root,a.logs,a.sensors,a.split,None,views=CAMERAS)
    excluded=set()
    if a.exclude:
        excluded={r["log_name"] for r in json.loads(Path(a.exclude).read_text())["records"]}
    records=[]; trajectories=[]
    tokens = sorted(loader.tokens)
    print(f"Exporting up to {a.limit} four-camera scenes from {len(tokens)} candidate tokens", flush=True)
    started = time.time()
    for index, token in enumerate(tokens, 1):
        raw=loader.scene_frames_dicts[token]; log_name=raw[3]["log_name"]
        if log_name in excluded: continue
        is_val=int(hashlib.sha256(log_name.encode()).hexdigest()[:8],16)%10==0
        if is_val != (a.role=="val"): continue
        scene=loader.get_scene_from_token(token); h=scene.scene_metadata.num_history_frames
        frames=scene.frames[:h+8]
        if len(frames)!=h+8: continue
        arrays=[]; timestamps=[]; camera_timestamps={c:[] for c in CAMERAS}
        for cam in CAMERAS:
            seq=[]
            for i,f in enumerate(frames):
                camera=getattr(f.cameras,cam)
                im=camera.image
                if im is None: break
                seq.append(np.asarray(Image.fromarray(im).convert("RGB").resize((256,256)),dtype=np.uint8))
                camera_timestamps[cam].append(raw[i]["cams"].get(cam.upper()))
            if len(seq)!=h+8: break
            arrays.append(np.stack(seq))
        if len(arrays)!=len(CAMERAS): continue
        image,state,text=legal_input(scene.get_agent_input())
        poses=scene.get_future_trajectory(8).poses.astype(np.float64)
        times=np.array([f.timestamp for f in frames],dtype=np.int64)
        np.savez_compressed(out/f"{token}.npz",views=np.stack(arrays),state=state,poses=poses,
            valid=np.ones(8,dtype=bool),timestamps=times,initial=np.stack([arrays[j][h] for j in range(4)]))
        records.append(dict(token=token,scene_token=scene.scene_metadata.scene_token,log_name=scene.scene_metadata.log_name,
            file=f"{token}.npz",views=list(CAMERAS),view_order="front,rear,left,right",timestamps=times.tolist(),
            camera_timestamps=camera_timestamps))
        trajectories.append(poses)
        if len(records) and (len(records) == 1 or len(records) % 10 == 0):
            elapsed = time.time() - started
            print(f"exported={len(records)}/{a.limit} scanned={index}/{len(tokens)} elapsed={elapsed:.1f}s token={token}", flush=True)
        if len(records)>=a.limit: break
    if not records: raise ValueError("No complete four-camera scenes selected")
    manifest=dict(navsim_commit=NAVSIM_COMMIT,split=a.split,role=a.role,dt=.5,horizon=4.,
        views=list(CAMERAS),view_order="front,rear,left,right",shape="[camera,time,channel,height,width]",records=records)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if a.role=="train": ActionStats.fit(trajectories,[r["token"] for r in records],a.role).save(out/"stats.json")
    print(json.dumps(dict(records=len(records),views=list(CAMERAS),output=str(out)),indent=2))

if __name__=="__main__": main()
