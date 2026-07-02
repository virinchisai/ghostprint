"""Step 2: extract prosody features (all manifests) and ECAPA baseline
embeddings (eval + calibration manifests).

Prosody features -> work/feat/prosody/<manifest>.pkl   {utt_key: {frames, stats}}
ECAPA embeddings -> work/feat/ecapa/<manifest>.npz     {utt_key: (192,) embedding}

utt_key = f"{utt_id}|{condition}" so clean/disguised versions coexist.
"""

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from ghostprint.config import WORK
from ghostprint.data import load_manifest
from ghostprint.features import extract

MAN = WORK / "manifests"
PROS = WORK / "feat" / "prosody"
ECAPA = WORK / "feat" / "ecapa"

ALL_MANIFESTS = sorted(p.stem for p in MAN.glob("*.jsonl"))
ECAPA_MANIFESTS = [m for m in ALL_MANIFESTS if m.startswith("eval_")]


def _prosody_job(row):
    return f"{row['utt_id']}|{row['condition']}", extract(row["path"])


def run_prosody():
    PROS.mkdir(parents=True, exist_ok=True)
    for name in ALL_MANIFESTS:
        out = PROS / f"{name}.pkl"
        if out.exists():
            print(f"prosody {name}: cached")
            continue
        rows = load_manifest(MAN / f"{name}.jsonl")
        feats = {}
        with Pool(4) as pool:
            for key, f in tqdm(pool.imap_unordered(_prosody_job, rows, chunksize=8),
                               total=len(rows), desc=f"prosody {name}"):
                feats[key] = f
        with open(out, "wb") as fh:
            pickle.dump(feats, fh)


def run_ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    import soundfile as sf

    ECAPA.mkdir(parents=True, exist_ok=True)
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(WORK / "pretrained" / "ecapa"),
        run_opts={"device": "cpu"},
    )
    for name in ECAPA_MANIFESTS:
        out = ECAPA / f"{name}.npz"
        if out.exists():
            print(f"ecapa {name}: cached")
            continue
        rows = load_manifest(MAN / f"{name}.jsonl")
        embs = {}
        for row in tqdm(rows, desc=f"ecapa {name}"):
            y, sr = sf.read(row["path"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            wav = torch.from_numpy(y).unsqueeze(0)
            with torch.no_grad():
                e = clf.encode_batch(wav).squeeze().numpy()
            embs[f"{row['utt_id']}|{row['condition']}"] = e / np.linalg.norm(e)
        np.savez(out, **embs)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "prosody"):
        run_prosody()
    if which in ("all", "ecapa"):
        run_ecapa()
    print("extract_features: done")
