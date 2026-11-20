"""Generate quantitative reports from actual recorded results."""

import json, platform, re, importlib.metadata as meta
from pathlib import Path
import numpy as np
from rusal_sim import ROOT, configuration
from rusal_sim.geometry import properties

c = configuration()
data = json.loads((ROOT / "results/reports/suite.json").read_text())
test_log = (ROOT / "results/logs/pytest.txt").read_text()
test_match = re.search(r"(\d+) passed in ([0-9.]+)s", test_log)
report = {
    "pytest": {
        "passed": int(test_match.group(1)) if test_match else None,
        "seconds": float(test_match.group(2)) if test_match else None,
        "has_failures": "FAILED" in test_log or " failed" in test_log,
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: meta.version(n)
            for n in ["mujoco", "numpy", "scipy", "trimesh", "matplotlib", "PyYAML", "pytest"]
        },
    },
    "geometry": properties(c["ingot"]),
    "cases": len(data),
    "passes": sum(r["success"] for r in data),
    "warnings": sum(bool(r["warnings"]) for r in data),
}
(ROOT / "results/reports/summary.json").write_text(json.dumps(report, indent=2))
lines = [
    "# Методика и фактические результаты",
    "",
    f"Выполнено {len(data)} детерминированных динамических сценария. Успешных: {report['passes']}; с предупреждениями MuJoCo: {report['warnings']}. Это сценарная выборка, не оценка вероятности надёжности.",
    "",
    "Критерий локального удержания: после подъёма высота центра >180 мм и смещение относительно инструмента <15 мм во всём интервале удержания; затем захват выключается. Проверки точности укладки 5 мм / 3° относятся к отдельному циклу. Время шага и решения: implicitfast, elliptic cone, multiccd, 100 итераций, шаг 1 мс; отдельные контрольные опыты 0,5 и 2 мс.",
    "",
    "| Сценарий | Масса, кг | Шаг, мс | Удержание | Макс. высота, м |",
    "|---|---:|---:|---|---:|",
]
for r in data:
    lines.append(
        f"| {r['case']} | {r['mass']} | {r['timestep'] * 1000:g} | {'да' if r['success'] else 'нет'} | {r['max_z']:.4f} |"
    )
lines += [
    "",
    "## Геометрия",
    "",
    f"Расчётный объём B: {report['geometry']['volume']:.6f} м³; масса при 2700 кг/м³: {report['geometry']['geometric_mass']:.3f} кг. При заданных 7,5 кг эффективная плотность {report['geometry']['effective_density']:.0f} кг/м³. Максимальная масса 15 кг требует независимой проверки состава и реальных габаритов.",
    "",
    "## Интерпретация",
    "",
    "Гладкая поверхность допускает вакуумное удержание в выбранных условиях. Рифлёная поверхность нарушает уплотнение и в этих опытах вакуум не удерживает груз. Это следствие явно принятой модели герметичности, а не измеренная характеристика реального изделия.",
    "",
    "При 15 кг и ограничении привода 300 Н на губку переход от μ=0,3 к μ=0,4 меняет исход механического опыта. Коэффициент трения и реальное нормальное усилие являются критическими параметрами. Комбинированный режим повышает доступность сценариев, но не гарантирует запас 3 и устойчивость к любому отказу.",
    "",
    "V2 в текущем представлении жёстких площадок теряет герметичность при подъёме. Требуется разрешение деформации пены и численная проверка; вывод о непригодности промышленной системы делать нельзя.",
    "",
    "Для одинаковых контрольных случаев при шагах 0,5, 1 и 2 мс исход удержания совпадает. Проверка не охватывает весь диапазон solref/solimp и не устраняет необходимость калибровки контакта.",
]
for name, file in [
    ("KUKA — независимый FK", "results/reports/kuka_fk.json"),
    ("KUKA — динамический цикл одной чушки", "results/reports/robot_trial.json"),
    ("Локальный цикл с соседями", "results/logs/pick_place.json"),
    ("Извлечение из осевшей навалки", "results/logs/heap_execution.json"),
    ("IK реальных контактных поз", "results/reports/heap_ik_audit.json"),
    ("ROS-мост", "results/reports/ros_smoke.json"),
    ("MoveIt — планирование и выполнение", "results/reports/moveit_smoke.json"),
    ("Пальцевая головка", "results/reports/rib_fingers.json"),
    ("Повторные вариации рёбер", "results/reports/robustness.json"),
    ("Чувствительность навалки", "results/reports/heap_numerical_sensitivity.json"),
    ("Вычислительная стоимость", "results/reports/heap_performance.json"),
]:
    p = ROOT / file
    if p.exists():
        lines += ["", f"## {name}", "", "```json", p.read_text(), "```"]
for n in [5, 50, 250]:
    p = ROOT / f"results/logs/heap{n}.json"
    if p.exists():
        lines += ["", f"## Навалка {n}", "", "```json", p.read_text(), "```"]
lines += [
    "",
    "## Что требуется измерить",
    "",
    "Распределение габаритов и массы; 3D-сканы рёбер и возможных поднутрений; трение по стали, резине и между чушками; усилие отрыва, сдвига и peel-момент; утечки в зависимости от пыли, шероховатости и температуры; жёсткость уплотнений; реальное распределение ориентаций и перекрытий; допуски паллеты и требования к маркировке. Частоту предварительного перемещения и минимальную конструктивную сложность текущая серия не определяет.",
]
(ROOT / "docs/experimental_results.md").write_text("\n".join(lines) + "\n")
# Robot coordinates and loads from the actual integrated KUKA trial.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = ROOT / "results/logs/robot_trial_samples.json"
if p.exists():
    rows = json.loads(p.read_text())
    t = [r["time"] for r in rows]
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i in range(6):
        axs[0].plot(t, [r["qpos"][i] for r in rows], label=f"A{i + 1}")
        axs[1].plot(t, [r["torque"][i] for r in rows], label=f"A{i + 1}")
    axs[0].set_ylabel("Joint position, rad")
    axs[1].set_ylabel("Actuator torque, Nm")
    axs[1].set_xlabel("Time, s")
    for ax in axs:
        ax.grid()
        ax.legend(ncol=6)
    fig.tight_layout()
    fig.savefig(ROOT / "results/plots/kuka_joints_and_torques.png")
    plt.close(fig)
print(report["cases"], "cases;", report["passes"], "passed")

heap_report = ROOT / "results/logs/heap_execution.json"
if heap_report.exists():
    r = json.loads(heap_report.read_text())
    attempts = r["attempts"]
    n = len(attempts)
    metrics = {
        label: sum(bool(a.get(flag, False)) for a in attempts) / n if n else None
        for label, flag in [
            ("grasp", "sealed"),
            ("extraction", "extraction_passed"),
            ("transport", "transport_passed"),
            ("placement", "placement_passed"),
            ("full_cycle", "success"),
        ]
    }
    (ROOT / "results/reports/heap_stage_rates.json").write_text(
        json.dumps(
            dict(
                attempts=n,
                rates=metrics,
                scope="conditional on accessible smooth candidates in one 10-body seed; not industrial success probabilities",
            ),
            indent=2,
        )
    )

    with (ROOT / "docs/experimental_results.md").open("a") as f:
        f.write("\n## Успешность этапов извлечения из навалки\n\n")
        f.write(
            "Два выбранных открытых гладких кандидата в одной навалке seed=42. Это условная частота по доступным кандидатам, не промышленная вероятность успеха.\n\n"
        )
        for label, value in metrics.items():
            f.write(
                f"- {label}: {value:.0%} ({sum(bool(a.get(dict(grasp='sealed', extraction='extraction_passed', transport='transport_passed', placement='placement_passed', full_cycle='success')[label], False)) for a in attempts)}/{n}).\n"
            )
        f.write(
            "\nПолный цикл не засчитан из-за максимального движения соседей 5,07 и 5,54 мм при пороге 5 мм. Фактически обе детали находятся на разных местах паллеты; занятые места резервируются независимо от итоговой оценки качества.\n"
        )
