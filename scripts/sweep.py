"""Formant dose-response sweep.

Varies ONE interpretable axis — vocal-tract-length / formant scaling, the
thing neural voice conversion restructures most — from mild (0.96) to
extreme (0.65), holding pitch roughly constant, and measures how each
system degrades. Produces the money figure: ECAPA's identification
falling off as formants are pushed, while GhostPrint stays flat because
it never relied on formants (timbre) in the first place.

Reuses the trained encoder, the cached clean enrollment centroids, and
evaluate.py's scoring functions. Only the disguised *probes* are new.
"""

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ghostprint.config import DISGUISE_PRESETS, SWEEP_PRESETS, RESULTS, WORK
from ghostprint.data import load_manifest, save_manifest
from ghostprint.disguise import apply_disguise
from ghostprint.features import extract
from ghostprint.model import ProsodyEncoder

import scripts.evaluate as ev   # scoring helpers + centroid construction

MAN = WORK / "manifests"
PROS = WORK / "feat" / "prosody"
ECAPA = WORK / "feat" / "ecapa"
WAV = WORK / "wav" / "eval"


def _disguise_job(job):
    in_path, out_path, preset = job
    if not Path(out_path).exists():
        apply_disguise(in_path, out_path, preset)
    return out_path


def _prosody_job(row):
    return f"{row['utt_id']}|{row['condition']}", extract(row["path"])


def build_preset(preset, probe_clean_rows):
    """Synthesize + feature-extract disguised probes for one preset."""
    man_path = MAN / f"eval_probe_{preset}.jsonl"
    rows = []
    jobs = []
    for r in probe_clean_rows:
        out = WAV / preset / f"{r['utt_id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        # need the ORIGINAL clean source path, not the clean-probe path
        src = r["path"]
        jobs.append((src, str(out), preset))
        nr = dict(r); nr["condition"] = preset; nr["path"] = str(out)
        rows.append(nr)
    save_manifest(rows, man_path)

    with Pool(4) as pool:
        list(pool.imap_unordered(_disguise_job, jobs, chunksize=8))

    # prosody features
    ppath = PROS / f"eval_probe_{preset}.pkl"
    if not ppath.exists():
        feats = {}
        with Pool(4) as pool:
            for key, f in pool.imap_unordered(_prosody_job, rows, chunksize=8):
                feats[key] = f
        with open(ppath, "wb") as fh:
            pickle.dump(feats, fh)

    # ECAPA embeddings
    epath = ECAPA / f"eval_probe_{preset}.npz"
    if not epath.exists():
        from speechbrain.inference.speaker import EncoderClassifier
        import soundfile as sf
        clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(WORK / "pretrained" / "ecapa"),
            run_opts={"device": "cpu"})
        embs = {}
        for r in rows:
            y, _ = sf.read(r["path"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            with torch.no_grad():
                e = clf.encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().numpy()
            embs[f"{r['utt_id']}|{preset}"] = e / np.linalg.norm(e)
        np.savez(epath, **embs)
    return man_path


def main():
    ckpt = torch.load(WORK / "ckpt" / "prosody_encoder.pt", map_location="cpu")
    enc = ProsodyEncoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()

    enroll_rows = load_manifest(MAN / "eval_enroll.jsonl")
    spk_order = sorted({r["speaker"] for r in enroll_rows})

    # centroids from cached clean enrollment (identical to evaluate.py)
    systems = {}
    for name, embed in [("ghostprint", lambda m: ev.prosody_embed(m, enc)),
                        ("ecapa", ev.ecapa_embed)]:
        enroll = embed("eval_enroll")
        cents = {}
        for s in spk_order:
            es = [enroll[f"{r['utt_id']}|clean"] for r in enroll_rows if r["speaker"] == s]
            c = np.mean(es, axis=0); cents[s] = c / np.linalg.norm(c)
        systems[name] = {"centroids": cents, "embed": embed}

    # the clean-probe manifest tells us WHICH original utterances are probes;
    # its 'path' points at the original LibriSpeech flac (condition=clean).
    probe_clean = load_manifest(MAN / "eval_probe_clean.jsonl")

    ratios = [DISGUISE_PRESETS[p][1] for p in SWEEP_PRESETS]
    curve = {"ratio": ratios, "ghostprint": [], "ecapa": []}

    # clean anchor (ratio 1.0)
    for name in ("ghostprint", "ecapa"):
        embs = systems[name]["embed"]("eval_probe_clean")
        keys = [f"{r['utt_id']}|clean" for r in probe_clean]
        E = [embs[k] for k in keys]
        Z, labels = ev.score_condition(E, [r["speaker"] for r in probe_clean],
                                       systems[name]["centroids"], spk_order)
        curve[name].append(ev.metrics_from(Z, labels))
    curve["ratio"] = [1.0] + ratios

    for preset in SWEEP_PRESETS:
        print(f"[sweep] building {preset} (formant {DISGUISE_PRESETS[preset][1]}) ...",
              flush=True)
        build_preset(preset, probe_clean)
        man = f"eval_probe_{preset}"
        rows = load_manifest(MAN / f"{man}.jsonl")
        keys = [f"{r['utt_id']}|{preset}" for r in rows]
        spks = [r["speaker"] for r in rows]
        for name in ("ghostprint", "ecapa"):
            embs = systems[name]["embed"](man)
            E = [embs[k] for k in keys]
            Z, labels = ev.score_condition(E, spks, systems[name]["centroids"], spk_order)
            m = ev.metrics_from(Z, labels)
            curve[name].append(m)
            print(f"    {name}: rank1={m['rank1']:.1f}%  EER={m['eer']:.1f}%", flush=True)

    with open(RESULTS / "sweep.json", "w") as f:
        json.dump(curve, f, indent=2)
    plot_sweep(curve)
    print("sweep: wrote results/sweep.json and results/formant_sweep.png", flush=True)


def plot_sweep(curve):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = curve["ratio"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, label, style in [("ecapa", "ECAPA voiceprint", "o-"),
                               ("ghostprint", "GhostPrint prosody", "s--")]:
        a1.plot(x, [m["rank1"] for m in curve[name]], style, label=label)
        a2.plot(x, [m["eer"] for m in curve[name]], style, label=label)
    a1.set_xlabel("formant scaling ratio (1.0 = undisguised → 0.65 = extreme)")
    a1.set_ylabel("Rank-1 identification (%)"); a1.set_title("Identification vs formant distortion")
    a1.invert_xaxis(); a1.legend(); a1.grid(alpha=0.3)
    a2.set_xlabel("formant scaling ratio (1.0 → 0.65)")
    a2.set_ylabel("EER (%)  — lower better"); a2.set_title("Verification error vs formant distortion")
    a2.invert_xaxis(); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle("Dose-response: voiceprint degrades under formant manipulation, prosody stays flat")
    fig.tight_layout()
    fig.savefig(RESULTS / "formant_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
