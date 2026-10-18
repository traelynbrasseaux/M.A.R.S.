"""Train on normal data only, then compare Mahalanobis and HDC on held-out clips."""

import argparse
import csv
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhm import DATA_DIR, RESULTS_DIR, device_ml
from mhm.capture.loader import load_capture
from mhm.datasets import mafaulda, mimii
from mhm.features.audio import AudioConfig, audio_features
from mhm.features.framing import frames
from mhm.features.vibration import VibrationConfig, vibration_features
from mhm.metrics import partial_auc, rates_at_threshold, roc_auc, roc_curve
from mhm.models import AnomalyModel
from mhm.models.hdc import HdcScorer
from mhm.models.mahalanobis import Mahalanobis

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three categorical slots, CVD-checked as a set


@dataclass
class Item:
    name: str
    group: str  # items sharing a group (one capture file) never straddle train and test
    abnormal: bool
    features: np.ndarray  # (windows, features)


Featurize = Callable[[str, np.ndarray, float], np.ndarray]


def python_features(kind: str, signal: np.ndarray, fs: float) -> np.ndarray:
    if kind == "accel":
        cfg = VibrationConfig(fs=fs)
        return vibration_features(frames(signal, cfg.size, cfg.hop), cfg)
    cfg = AudioConfig(fs=fs)
    return audio_features(frames(signal, cfg.size, cfg.hop), cfg)


@dataclass
class DeviceModel:
    """One score column of the node's C++ models, trained in its two phases on halves of train."""

    column: int
    stats: np.ndarray = field(init=False, repr=False)
    bundle: np.ndarray = field(init=False, repr=False)

    def fit(self, normal_windows: np.ndarray) -> Self:
        half = len(normal_windows) // 2
        self.stats, self.bundle = normal_windows[:half], normal_windows[half:]
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return device_ml.scores(self.stats, self.bundle, windows)[:, self.column]


@dataclass
class Result:
    model: str
    auc: float
    pauc: float
    threshold: float
    detection_rate: float
    false_alarm_rate: float
    labels: np.ndarray
    scores: np.ndarray


def mimii_items(args: argparse.Namespace, featurize: Featurize) -> Iterator[Item]:
    root = args.root or DATA_DIR / "mimii" / f"{args.snr}dB"
    for clip in mimii.list_clips(root, args.machine, args.machine_id):
        x, fs = mimii.load_wav(clip.path)
        if fs != AudioConfig().fs:
            raise ValueError(f"{clip.path} is {fs} Hz, expected {AudioConfig().fs:g}")
        name = f"{clip.path.parent.name}/{clip.path.name}"
        yield Item(name, name, clip.abnormal, featurize("audio", x, fs))


def mafaulda_items(args: argparse.Namespace, featurize: Featurize) -> Iterator[Item]:
    root = args.root or DATA_DIR / "mafaulda"
    for clip in mafaulda.list_clips(root, args.fault):
        x, fs = mafaulda.load_vibration(clip.path)
        name = clip.path.relative_to(root).as_posix()
        yield Item(name, name, clip.abnormal, featurize("accel", x, fs))


def capture_items(args: argparse.Namespace, featurize: Featurize) -> Iterator[Item]:
    """Cut each capture in captures.csv into fixed-length segments labelled by condition."""
    root = args.root or DATA_DIR / "captures"
    with (root / "captures.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        capture = load_capture(root / row["file"])
        stream = capture.accel if args.stream == "accel" else capture.audio
        length = int(args.segment_s * stream.fs)
        for i, segment in enumerate(stream.segments()):
            signal = segment if args.stream == "accel" else segment[:, 0]
            for j in range(len(signal) // length):
                piece = signal[j * length : (j + 1) * length]
                features = featurize(args.stream, piece, stream.fs)
                yield Item(
                    f"{row['file']}:{i}:{j}", row["file"], row["condition"] != "normal", features
                )


def split(items: list[Item], rng: np.random.Generator) -> tuple[list[Item], list[Item], list[Item]]:
    """Deal whole normal groups into test, calibration, and train; no run straddles two."""
    abnormal = [i for i in items if i.abnormal]
    groups: dict[str, list[Item]] = {}
    for item in items:
        if not item.abnormal:
            groups.setdefault(item.group, []).append(item)
    order = list(groups)
    rng.shuffle(order)

    def take(target: int) -> list[Item]:
        taken: list[Item] = []
        while order and (not taken or len(taken) + len(groups[order[-1]]) <= target):
            taken += groups[order.pop()]
        return taken

    n_normal = sum(len(g) for g in groups.values())
    test = take(min(len(abnormal), n_normal // 2))
    calibration = take(round(0.2 * (n_normal - len(test))))
    train = [item for g in order for item in groups[g]]
    if not abnormal or not train:
        raise SystemExit("need abnormal data and at least three normal groups (e.g. captures)")
    return train, calibration, test + abnormal


def item_scores(model: AnomalyModel, items: list[Item]) -> np.ndarray:
    """Mean window score per item, the DCASE way of turning frame scores into a clip score."""
    window_scores = model.score(np.concatenate([i.features for i in items]))
    bounds = np.cumsum([len(i.features) for i in items])[:-1]
    return np.array([s.mean() for s in np.split(window_scores, bounds)])


def evaluate(
    models: dict[str, AnomalyModel],
    train: list[Item],
    calibration: list[Item],
    test: list[Item],
    max_fpr: float,
    quantile: float,
) -> list[Result]:
    train_x = np.concatenate([i.features for i in train])
    labels = np.array([i.abnormal for i in test])
    results = []
    for name, model in models.items():
        model.fit(train_x)
        threshold = float(np.quantile(item_scores(model, calibration), quantile))
        scores = item_scores(model, test)
        detection, false_alarm = rates_at_threshold(labels, scores, threshold)
        results.append(
            Result(
                name,
                roc_auc(labels, scores),
                partial_auc(labels, scores, max_fpr),
                threshold,
                detection,
                false_alarm,
                labels,
                scores,
            )
        )
    return results


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc(results: list[Result], max_fpr: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.4), facecolor=SURFACE)
    style(ax)
    ax.axvspan(0, max_fpr, color=GRID, alpha=0.6, linewidth=0)
    ax.text(max_fpr / 2, 0.03, f"pAUC\nFPR<{max_fpr:g}", ha="center", color=MUTED, fontsize=8)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for r, color in zip(results, SERIES, strict=False):
        fpr, tpr = roc_curve(r.labels, r.scores)
        label = f"{r.model}   AUC {r.auc:.3f}   pAUC {r.pauc:.3f}"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_title(title, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_scores(results: list[Result], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8), facecolor=SURFACE)
    for ax, r, color in zip(np.atleast_1d(axes), results, SERIES, strict=False):
        style(ax)
        lo, hi = r.scores.min(), r.scores.max()
        if lo > 0 and hi / lo > 100:
            bins = np.geomspace(lo, hi, 31)
            ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 31)
        for abnormal, c, label in [(False, MUTED, "normal"), (True, color, "abnormal")]:
            s = r.scores[r.labels == abnormal]
            ax.hist(s, bins=bins, histtype="stepfilled", color=c, alpha=0.3, linewidth=0)
            ax.hist(s, bins=bins, histtype="step", color=c, linewidth=2, label=label)
        ax.axvline(r.threshold, color=INK, linewidth=1)
        ax.text(r.threshold, ax.get_ylim()[1] * 0.97, " threshold", color=INK, fontsize=8, va="top")
        ax.set_title(
            f"{r.model}\ndetection {r.detection_rate:.2f}, false alarm {r.false_alarm_rate:.2f}",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("clip score", color=INK_2)
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=8, loc="best")
    np.atleast_1d(axes)[0].set_ylabel("clips", color=INK_2)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["mimii", "mafaulda", "captures"])
    parser.add_argument("--root", type=Path, help="dataset folder; defaults under data/")
    parser.add_argument("--machine", default="pump", help="mimii machine type")
    parser.add_argument("--machine-id", default="id_00", help="mimii machine ID")
    parser.add_argument("--snr", default="6", help="mimii SNR in dB")
    parser.add_argument("--fault", default="imbalance", help="mafaulda fault folder")
    parser.add_argument("--stream", default="accel", choices=["accel", "audio"], help="captures")
    parser.add_argument("--segment-s", type=float, default=5.0, help="captures segment length")
    parser.add_argument("--hdc-dim", type=int, default=4096)
    parser.add_argument("--max-fpr", type=float, default=0.1, help="pAUC FPR limit")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0, help="train and test split seed")
    parser.add_argument(
        "--backend",
        default="python",
        choices=["python", "device"],
        help="device runs the node's C++ features and models through the native build",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    featurize: Featurize = python_features
    if args.backend == "device":
        device_ml.build()
        featurize = device_ml.features
    match args.dataset:
        case "mimii":
            items = list(mimii_items(args, featurize))
            tag = f"mimii_{args.machine}_{args.machine_id}_{args.snr}dB"
        case "mafaulda":
            items = list(mafaulda_items(args, featurize))
            tag = f"mafaulda_{args.fault}"
        case "captures":
            items = list(capture_items(args, featurize))
            tag = f"captures_{args.stream}"
    if args.backend == "device":
        tag += "_device"
    if not items:
        raise SystemExit("no data found; run the download script or a capture first")

    train, calibration, test = split(items, np.random.default_rng(args.seed))
    n_abnormal = sum(i.abnormal for i in test)
    print(
        f"{tag}: train {len(train)} normal, calibration {len(calibration)} normal,"
        f" test {len(test) - n_abnormal} normal + {n_abnormal} abnormal"
    )
    models: dict[str, AnomalyModel] = {
        "mahalanobis": Mahalanobis(),
        "hdc-binary": HdcScorer(dim=args.hdc_dim, mode="binary"),
        "hdc-bipolar": HdcScorer(dim=args.hdc_dim, mode="bipolar"),
    }
    if args.backend == "device":
        models = {name: DeviceModel(i) for i, name in enumerate(models)}
    results = evaluate(models, train, calibration, test, args.max_fpr, args.threshold_quantile)

    header = f"{'model':<12} {'AUC':>6} {'pAUC':>6} {'detect':>7} {'false alarm':>12}"
    print(header)
    for r in results:
        print(
            f"{r.model:<12} {r.auc:6.3f} {r.pauc:6.3f}"
            f" {r.detection_rate:7.2f} {r.false_alarm_rate:12.2f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{tag}_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "auc", "pauc", "threshold", "detection_rate", "false_alarm_rate"])
        for r in results:
            writer.writerow(
                [r.model, r.auc, r.pauc, r.threshold, r.detection_rate, r.false_alarm_rate]
            )
    title = f"{tag}  (threshold at the {args.threshold_quantile:g} quantile of calibration normals)"
    plot_roc(results, args.max_fpr, tag, args.out / f"{tag}_roc.png")
    plot_scores(results, title, args.out / f"{tag}_scores.png")
    print(f"wrote {tag}_metrics.csv, {tag}_roc.png, {tag}_scores.png to {args.out}")


if __name__ == "__main__":
    main()
