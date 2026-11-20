"""Package the custom hybrid experiment without changing the two baseline videos."""

import hashlib
import json
import zipfile
from pathlib import Path
from package_gripper_videos import package, DEST, ROOT

report = json.loads((DEST / "hybrid_run.json").read_text())
video_name = "04_HYBRID_ALL5.mp4" if report["complete"] else "03_HYBRID_FOAM_JAWS.mp4"
r = package(
    "hybrid", "Гибрид: вакуум + опускаемые губки", "configs/hybrid_foam_jaws.yaml", video_name
)
audit = DEST / "hybrid_run_clearance.json"
if audit.exists():
    r["tool_robot_clearance_audit"] = json.loads(audit.read_text())
r["initial_state_sha256"] = hashlib.sha256(
    (ROOT / "results/logs/compact5_seed42.npz").read_bytes()
).hexdigest()
(DEST / "hybrid_summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
text = [
    "ГИБРИДНЫЙ ЗАХВАТ — ИССЛЕДОВАТЕЛЬСКАЯ ФИЗИЧЕСКАЯ СИМУЛЯЦИЯ",
    "",
    "Пять свободных чушек по 7,5 кг, та же навалка seed 42. Укладка на паллету с зазорами 40 мм.",
    "Вакуум поднимает деталь на 220 мм, затем открытые губки опускаются и закрываются.",
    "После захвата вакуум физически выключается на 1 секунду. Проверяются боковые контакты, отсутствие падения и смещение менее 10 мм.",
    "При успешной проверке перенос выполняется губками с выключенным вакуумом.",
    "Если вакуум не сработал, допускается прямой механический захват при свободном проходе губок; этот случай отмечается отдельно.",
    "Перед втягиванием пальцы должны раскрыться. Все перемещения и удержание рассчитываются динамикой, без прикрепления чушки к инструменту.",
    "При размещении механический захват отпускает деталь в 25 мм над опорой; далее она оседает под действием тяжести.",
    "",
    f"Перенесено {r['transferred']} из 5; строгий критерий качества {r['quality_passes']}; лимиты робота соблюдены: {r['robot_limits_pass']}.",
    f"Максимальная ошибка установки перенесённых деталей: {r['maximum_placement_error_mm']:.2f} мм.",
    "",
    f"Причина завершения: {'все пять деталей размещены' if report['complete'] else r['stop_reason']}",
    "Проверки передачи нагрузки с отключением вакуума:",
]
for a in r["vacuum_off_handoffs"]:
    text.append(
        f"{a['selected']}: удержание {a['passed']}; смещение за 1 с {a['drift_mm']:.3f} мм."
    )
text += [
    "",
    "Это собственная комбинация приблизительных моделей, а не существующее серийное изделие или цифровой двойник производителя.",
    "Восприятие, планирование, регуляторы и пневмоинтерфейсы заменены моками. Тела, контакты, силы и ограничения робота физические.",
    *r["assumptions"],
    "",
    "Источники исходных параметров:",
    *r["sources"],
    "",
    "Запуск из <repository-root>:",
    "./run_sim.sh --config configs/hybrid_foam_jaws.yaml --scenario compact --count 5 --seed 42 --mode hybrid",
    (
        "Полный цикл завершён: все пять деталей перенесены."
        if report["complete"]
        else "Полный цикл не завершён; фактическая причина остановки приведена выше."
    ),
    "Конечная заставка и искусственная пауза в конце видео не добавляются.",
    "",
    "MP4 / H.264 / 960×720 / 30 кадров/с / 1× / без звука. Крупный план воспроизводит записанные состояния с шагом 20 мс.",
    "Исходные JSON/CSV/XML/NPZ: results/share/gripper_comparison/hybrid_run.*",
]
(DEST / "HYBRID_README_RU.txt").write_text("\n".join(text))
files = [video_name, "hybrid_summary.json", "HYBRID_README_RU.txt"]
archive_name = "hybrid_all5.zip" if report["complete"] else "hybrid_gripper.zip"
with zipfile.ZipFile(DEST.parent / archive_name, "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(DEST / name, arcname=name)
print(json.dumps(r, ensure_ascii=False, indent=2))
