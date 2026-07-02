"""Step 4: open-set re-identification evaluation on 40 UNSEEN speakers.

Protocol (mirrors a fraudster watchlist):
  - enroll: 5 clean utterances/speaker -> centroid ("the fraudster's file")
  - probes: 12 utterances/speaker, tested clean and under each disguise
  - per system (ECAPA baseline / GhostPrint prosody / score fusion):
      EER, rank-1 identification, TPR @ 1% FAR (watchlist hit-rate)
  - scores are probe-side z-normalized against the non-target cohort
    before fusion (standard s-norm; no eval-side tuning, fusion = equal
    weights on normalized scores).

Writes results/metrics.json, results/summary.md, results/eer_by_condition.png,
results/tpr_by_condition.png and results/calibration.json (for demo.py).
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ghostprint.config import DISGUISE_PRESETS, RESULTS, WORK
from ghostprint.data import load_manifest
from ghostprint.model import ProsodyEncoder

MAN = WORK / "manifests"
PROS = WORK / "feat" / "prosody"
ECAPA = WORK / "feat" / "ecapa"
CONDITIONS = ["clean"] + list(DISGUISE_PRESETS)


# ---------------------------------------------------------------- embeddings
def prosody_embed(manifest: str, enc: ProsodyEncoder) -> dict[str, np.ndarray]:
    with open(PROS / f"{manifest}.pkl", "rb") as f:
        feats = pickle.load(f)
    out = {}
    with torch.no_grad():
        for key, ft in feats.items():
            frames = torch.from_numpy(ft["frames"]).unsqueeze(0)
            lengths = torch.tensor([frames.size(1)])
            stats = torch.from_numpy(ft["stats"]).unsqueeze(0)
            out[key] = enc(frames, lengths, stats).squeeze(0).numpy()
    return out


def ecapa_embed(manifest: str) -> dict[str, np.ndarray]:
    npz = np.load(ECAPA / f"{manifest}.npz")
    return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------- metrics
def eer(target: np.ndarray, nontarget: np.ndarray) -> float:
    thr = np.sort(np.concatenate([target, nontarget]))
    fnr = np.array([(target < t).mean() for t in thr])
    fpr = np.array([(nontarget >= t).mean() for t in thr])
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2 * 100)


def tpr_at_far(target: np.ndarray, nontarget: np.ndarray, far=0.01) -> float:
    thr = np.quantile(nontarget, 1 - far)
    return float((target >= thr).mean() * 100)


def znorm_scores(S: np.ndarray) -> np.ndarray:
    """Probe-side cohort z-norm: for each probe, normalize its score to
    centroid j by the stats of its scores to all other centroids."""
    Z = np.zeros_like(S)
    n = S.shape[1]
    for j in range(n):
        cohort = np.delete(S, j, axis=1)
        Z[:, j] = (S[:, j] - cohort.mean(axis=1)) / (cohort.std(axis=1) + 1e-6)
    return Z


def score_condition(probe_embs, probe_spks, centroids, spk_order):
    S = np.stack([[e @ centroids[s] for s in spk_order] for e in probe_embs])
    Z = znorm_scores(S)
    labels = np.array([spk_order.index(s) for s in probe_spks])
    return Z, labels


def metrics_from(Z, labels):
    tgt = Z[np.arange(len(labels)), labels]
    mask = np.ones_like(Z, dtype=bool)
    mask[np.arange(len(labels)), labels] = False
    non = Z[mask]
    rank1 = float((Z.argmax(axis=1) == labels).mean() * 100)
    return {"eer": round(eer(tgt, non), 2),
            "rank1": round(rank1, 2),
            "tpr_at_1far": round(tpr_at_far(tgt, non), 2)}


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(exist_ok=True)
    ckpt = torch.load(WORK / "ckpt" / "prosody_encoder.pt", map_location="cpu")
    enc = ProsodyEncoder()
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    enroll_rows = load_manifest(MAN / "eval_enroll.jsonl")
    spk_order = sorted({r["speaker"] for r in enroll_rows})

    systems = {}
    for name, embed in [("ghostprint", lambda m: prosody_embed(m, enc)),
                        ("ecapa", ecapa_embed)]:
        enroll = embed("eval_enroll")
        cents = {}
        for s in spk_order:
            es = [enroll[f"{r['utt_id']}|clean"] for r in enroll_rows
                  if r["speaker"] == s]
            c = np.mean(es, axis=0)
            cents[s] = c / np.linalg.norm(c)
        systems[name] = {"centroids": cents, "embed": embed}

    results = {name: {} for name in list(systems) + ["fusion"]}
    for cond in CONDITIONS:
        manifest = "eval_probe_clean" if cond == "clean" else f"eval_probe_{cond}"
        rows = load_manifest(MAN / f"{manifest}.jsonl")
        keys = [f"{r['utt_id']}|{r['condition']}" for r in rows]
        spks = [r["speaker"] for r in rows]

        Zs = {}
        for name, sysd in systems.items():
            embs = sysd["embed"](manifest)
            E = [embs[k] for k in keys]
            Z, labels = score_condition(E, spks, sysd["centroids"], spk_order)
            Zs[name] = Z
            results[name][cond] = metrics_from(Z, labels)
        Zf = (Zs["ecapa"] + Zs["ghostprint"]) / 2.0
        results["fusion"][cond] = metrics_from(Zf, labels)
        print(f"[{cond}] " + "  ".join(
            f"{n}: EER {results[n][cond]['eer']:.1f}%" for n in results))

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    write_summary(results)
    plot(results)
    print(f"wrote {RESULTS}/metrics.json, summary.md, plots")


def write_summary(results):
    lines = ["# GhostPrint evaluation — 40 unseen speakers (LibriSpeech test-clean)",
             "",
             "Watchlist protocol: 5-utt clean enrollment, 12 probes/spk.",
             "'hard' disguise preset was never seen in training.", ""]
    for metric, title in [("eer", "EER % (lower better)"),
                          ("rank1", "Rank-1 identification % (higher better)"),
                          ("tpr_at_1far", "Watchlist hit-rate % @ 1% FAR (higher better)")]:
        lines += [f"## {title}", "",
                  "| condition | ECAPA (voiceprint) | GhostPrint (prosody) | Fusion |",
                  "|---|---|---|---|"]
        for cond in CONDITIONS:
            lines.append(
                f"| {cond} | {results['ecapa'][cond][metric]} "
                f"| {results['ghostprint'][cond][metric]} "
                f"| {results['fusion'][cond][metric]} |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel, fname in [
            ("eer", "EER (%)  — lower is better", "eer_by_condition.png"),
            ("tpr_at_1far", "Watchlist hit-rate @ 1% FAR (%)", "tpr_by_condition.png")]:
        x = np.arange(len(CONDITIONS))
        w = 0.27
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, (name, label) in enumerate([
                ("ecapa", "ECAPA voiceprint (baseline)"),
                ("ghostprint", "GhostPrint prosody"),
                ("fusion", "Fusion")]):
            vals = [results[name][c][metric] for c in CONDITIONS]
            ax.bar(x + (i - 1) * w, vals, w, label=label)
        ax.set_xticks(x, CONDITIONS)
        ax.set_ylabel(ylabel)
        ax.set_title("Repeat-speaker re-identification under voice disguise")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
