"""Render saved physical states. No integration, title cards or added stills."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--speed", type=float, default=2.0)
    args = p.parse_args()
    if args.speed <= 0:
        p.error("speed must be positive")
    report = json.loads(args.source.with_suffix(".json").read_text())
    with args.source.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    model = mujoco.MjModel.from_xml_path(str(args.source.with_suffix(".xml").resolve()))
    data = mujoco.MjData(model)
    times = np.array([float(r["time"]) for r in rows])
    states = np.array([[float(r[f"qpos_{i}"]) for i in range(model.nq)] for r in rows])
    start = float(times[0])
    end = float(times[-1])
    fps = 30
    frames = int(np.floor((end - start) / args.speed * fps)) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overview = mujoco.Renderer(model, 720, 960)
    detail = mujoco.Renderer(model, 240, 320)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.1, -0.1, 0.35]
    camera.distance = 4.8
    camera.azimuth = 125
    camera.elevation = -40
    close = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(close)
    close.distance = 0.95
    close.elevation = -25
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    writer = imageio.get_writer(
        str(args.output),
        fps=fps,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
    )
    tool = model.body("tool").id
    jaw_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_motor_" + side)
        for side in ("left", "right")
    ]
    try:
        for frame in range(frames):
            t = start + frame * args.speed / fps
            index = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
            if index and abs(times[index - 1] - t) < abs(times[index] - t):
                index -= 1
            row = rows[index]
            data.qpos[:] = states[index]
            mujoco.mj_forward(model, data)
            overview.update_scene(data, camera=camera)
            im = Image.fromarray(overview.render())
            close.lookat[:] = data.xpos[tool] + [0, 0, -0.015]
            R = data.xmat[tool].reshape(3, 3)
            close.azimuth = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))) + 35
            detail.update_scene(data, camera=close)
            im.paste(Image.fromarray(detail.render()), (12, 460))
            draw = ImageDraw.Draw(im)
            draw.rectangle((11, 459, 332, 700), outline=(90, 180, 230), width=2)
            item = report["configuration"].get("ingots", {}).get(row["selected"])
            label = f"{row['selected']} | {row['phase']} | t={float(row['time']):.1f} s | x{args.speed:g}"
            if item:
                label += f" | {item['mass']:.2f} kg"
            if all(a >= 0 for a in jaw_ids):
                label += (
                    " | jaw "
                    + ("/".join(f"{float(row[f'actuator_force_{a}']):+.0f}" for a in jaw_ids))
                    + " N"
                )
            draw.rectangle((0, 0, 960, 31), fill=(10, 20, 30))
            draw.text((12, 7), label, font=font, fill="white")
            writer.append_data(np.asarray(im))
            if frame % 300 == 0:
                print(f"{frame}/{frames}", flush=True)
    finally:
        writer.close()
        overview.close()
        detail.close()
    metadata = dict(
        source=str(args.source),
        video=str(args.output),
        replay="nearest recorded qpos, 50 Hz; no new dynamics",
        speed=args.speed,
        fps=fps,
        frames=frames,
        duration_s=frames / fps,
        source_time_range_s=[start, end],
        beginning_title_card=False,
        ending_title_card=False,
        added_still_frames=0,
        complete=report["complete"],
        transferred=report["transferred"],
        quality_passes=report["quality_passes"],
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
