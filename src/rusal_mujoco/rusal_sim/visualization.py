"""Read-only overlays for rendered dynamics; never changes simulation state."""

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont


def status_frame(frame, m, d, vacuum, state, mode, selected="ingot", note=""):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    jaws = [
        d.qpos[m.joint("jaw_" + side).qposadr[0]] * 1000
        for side in ["left", "right"]
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jaw_" + side) >= 0
    ]
    lines = [
        mode,
        f"{state} | t={d.time:.2f} s | selected: {selected}",
        f"Vacuum: {vacuum.mean_pressure / 1000 if vacuum else 0:.1f} kPa | seal: {vacuum.mean_quality if vacuum else 0:.2f} | capacity: {sum(vacuum.capacity) if vacuum else 0:.1f} N",
        f"Jaw travel, mm: {' / '.join(f'{q:.1f}' for q in jaws) or 'no jaws'} | free loads, contact dynamics",
    ]
    draw.rectangle([8, 8, img.width - 8, 116], fill=(0, 0, 0, 175))
    for i, line in enumerate(lines):
        draw.text((17, 13 + 25 * i), line, font=font, fill=(255, 255, 255, 255))
    if note:
        import textwrap

        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        wrapped = textwrap.wrap(note, 110)
        y = img.height - 12 - 20 * len(wrapped)
        draw.rectangle([8, y - 5, img.width - 8, img.height - 8], fill=(0, 0, 0, 195))
        for i, line in enumerate(wrapped):
            draw.text((17, y + 20 * i), line, font=small, fill=(255, 220, 160, 255))
    return np.asarray(img)
