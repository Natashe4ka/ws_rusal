"""Make shareable videos from recorded physics, with a close-up state replay.

The inset uses saved qpos samples (50 Hz), never new integration or a fabricated
attachment. The overview is the original 30 fps recording. Nearest samples are
used, so the inset time quantization is at most 20 ms.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/share/gripper_comparison"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def package(name, title, config, output):
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes()
        + b"".join(
            (DEST / f"{name}_{suffix}").read_bytes()
            for suffix in ["run.json", "run.csv", "run.xml", "raw.mp4"]
        )
    ).hexdigest()
    metadata_path = DEST / f"{name}_video_metadata.json"
    if metadata_path.exists() and (DEST / output).exists():
        cached = json.loads(metadata_path.read_text())
        if cached.get("input_sha256") == fingerprint:
            print("Reusing verified recording package", name, flush=True)
            return cached
    print("Packaging", name, flush=True)
    report = json.loads((DEST / f"{name}_run.json").read_text())
    with (DEST / f"{name}_run.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(r["time"]) for r in rows])
    m = mujoco.MjModel.from_xml_path(str(DEST / f"{name}_run.xml"))
    data = mujoco.MjData(m)
    states = np.array([[float(r[f"qpos_{i}"]) for i in range(m.nq)] for r in rows])
    renderer = mujoco.Renderer(m, 240, 320)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.9
    cam.elevation = -25
    tool = m.body("tool").id
    attempts = report["attempts"]
    max_error = max(
        (a["position_error_m"] * 1000 for a in attempts if a.get("physically_deposited")), default=0
    )
    reasons = list(dict.fromkeys(a.get("failure", "") for a in attempts if a.get("failure")))
    writer = imageio.get_writer(
        str(DEST / output),
        fps=30,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
    )
    reader = imageio.get_reader(str(DEST / f"{name}_raw.mp4"))
    font = ImageFont.truetype(FONT, 14)
    try:
        prefix = round(report.get("video_initial_still_s", 1.0) * 30)
        dynamic_frames = int(np.ceil(report["simulation_time_s"] * 30 - 1e-8))
        for fi, frame in enumerate(reader):
            if fi < prefix:
                continue
            if fi >= prefix + dynamic_frames:
                break
            t = (fi - prefix) / 30
            ix = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
            if ix > 0 and abs(times[ix - 1] - t) < abs(times[ix] - t):
                ix -= 1
            data.qpos[:] = states[ix]
            mujoco.mj_forward(m, data)
            cam.lookat[:] = data.xpos[tool] + [0, 0, -0.015]
            R = data.xmat[tool].reshape(3, 3)
            cam.azimuth = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))) + 35
            renderer.update_scene(data, camera=cam)
            inset = Image.fromarray(renderer.render()).resize((360, 270))
            im = Image.fromarray(frame)
            im.paste(inset, (12, 350))
            dr = ImageDraw.Draw(im)
            dr.rectangle((11, 327, 373, 349), fill=(10, 20, 30))
            dr.rectangle((11, 349, 373, 622), outline=(90, 180, 230), width=2)
            dr.text((18, 330), "Крупный план · записанные состояния", font=font, fill="white")
            writer.append_data(np.asarray(im))
            if fi % 600 == 0:
                print(name, "frame", fi, flush=True)
    finally:
        reader.close()
        writer.close()
        renderer.close()
    result = dict(
        gripper=title,
        video=output,
        config=config,
        input_sha256=fingerprint,
        transferred=report["transferred"],
        remaining=report["remaining"],
        quality_passes=report["quality_passes"],
        complete=report["complete"],
        robot_collisions=report["robot_collisions"],
        final_placement_audit=report["final_placement_audit"],
        beginning_title_card=False,
        ending_title_card=False,
        simulation_time_s=report["simulation_time_s"],
        maximum_placement_error_mm=max_error,
        robot_limits_pass=report["robot_limits_pass"],
        stop_reason=report["stop_reason"],
        failure_details=reasons,
        max_speed_ratio=report["max_speed_ratio"],
        max_effort_ratio=report["max_effort_ratio"],
        warnings=report["warnings"],
        assumptions=report["configuration"]["gripper"]["assumptions"],
        transferred_ingots=[
            dict(
                selected=a["selected"],
                surface=a["surface"],
                position_error_mm=1000 * a["position_error_m"],
            )
            for a in attempts
            if a.get("physically_deposited")
        ],
        vacuum_off_handoffs=[
            dict(
                selected=a["selected"],
                passed=a["vacuum_off_hold_passed"],
                drift_mm=1000 * a["vacuum_off_hold_drift_m"],
            )
            for a in attempts
            if "vacuum_off_hold_passed" in a
        ],
        sources=report["configuration"]["gripper"]["sources"],
    )
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    results = [
        package("piab", "Piab Kenos KVG150", "configs/piab_kvg150.yaml", "01_PIAB_KVG150.mp4"),
        package(
            "schunk",
            "SCHUNK PGN-plus-P 200-1",
            "configs/schunk_pgn200.yaml",
            "02_SCHUNK_PGN200.mp4",
        ),
    ]
    summary = dict(
        scope="Catalog-based research approximations, not manufacturer-validated digital twins",
        initial_heap="results/logs/compact5_seed42",
        mass_per_ingot_kg=7.5,
        pallet_gaps_mm=[40, 40],
        comparison="Same settled initial state; 1 deterministic seed, not an industrial reliability estimate",
        results=results,
    )
    summary["initial_state_sha256"] = hashlib.sha256(
        (ROOT / "results/logs/compact5_seed42.npz").read_bytes()
    ).hexdigest()
    (DEST / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        "СРАВНЕНИЕ ЗАХВАТОВ В ФИЗИЧЕСКОЙ СИМУЛЯЦИИ",
        "",
        "01_PIAB_KVG150.mp4 — вакуумный вариант, приближение пены упругими плитками.",
        "02_SCHUNK_PGN200.mp4 — серийный механический привод с условными плоскими пальцами.",
        "MP4 / H.264 / 960×720 / 30 кадров/с / скорость 1× / без звука.",
        "Вставка крупного плана воспроизводит записанные qpos с шагом 20 мс, а не новый расчёт динамики.",
        "",
        "Оба расчёта стартуют с одной физически осевшей навалки из 5 деталей по 7,5 кг.",
        "Паллета 1200×800 мм, 5 позиций в один ряд, зазоры 40 мм. Механическое выравнивание после укладки не моделируется.",
        "Робот KUKA KR 50 R2500 с реальными геометрией, инерциями и ограниченными усилиями приводов.",
        "Чушки свободны: без weld/mocap/телепортации при исполнении. Восприятие, планирование и контроллеры — моки.",
        "Модели захватов приближённые: не CAD/цифровые двойники производителей и не доказательство пригодности для закупки.",
        "",
    ]
    for r in results:
        lines += [
            r["gripper"],
            f"Перенесено {r['transferred']}/5; строгий критерий качества {r['quality_passes']}; максимальная ошибка {r['maximum_placement_error_mm']:.2f} мм.",
            f"Лимиты соблюдены: {r['robot_limits_pass']}; предупреждения движка: {sum(r['warnings'])}.",
            *r["assumptions"],
            *r["sources"],
            "",
        ]
    lines += [
        "Запуск из <repository-root>:",
        "./run_sim.sh --config configs/piab_kvg150.yaml --scenario compact --count 5 --seed 42 --mode vacuum",
        "./run_sim.sh --config configs/schunk_pgn200.yaml --scenario compact --count 5 --seed 42 --mode mechanical",
        "Полные JSON/CSV/XML/NPZ и исходные видео сохранены рядом с готовыми роликами.",
    ]
    (DEST / "README_RU.txt").write_text("\n".join(lines))
    include = ["01_PIAB_KVG150.mp4", "02_SCHUNK_PGN200.mp4", "README_RU.txt", "comparison.json"]
    manifest = {name: hashlib.sha256((DEST / name).read_bytes()).hexdigest() for name in include}
    (DEST / "SHA256.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(
        DEST.parent / "gripper_comparison.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in include + ["SHA256.json"]:
            archive.write(DEST / name, arcname=name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
