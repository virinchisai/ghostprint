"""Investigator demo: are these two calls the same person?

    python demo.py call_a.wav call_b.wav

Prints per-channel similarity (timbre voiceprint vs clone-resistant
prosody fingerprint) and a fused verdict — the point being that when
the two channels DISAGREE (voiceprint says no, prosody says yes),
that's the signature of a repeat fraudster hiding behind a new voice.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from ghostprint.config import WORK
from ghostprint.features import extract
from ghostprint.model import ProsodyEncoder


def prosody_emb(enc, path):
    ft = extract(path)
    frames = torch.from_numpy(ft["frames"]).unsqueeze(0)
    stats = torch.from_numpy(ft["stats"]).unsqueeze(0)
    with torch.no_grad():
        return enc(frames, torch.tensor([frames.size(1)]), stats).squeeze(0).numpy()


def ecapa_emb(clf, path):
    import soundfile as sf
    y, _ = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    with torch.no_grad():
        e = clf.encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_a")
    ap.add_argument("wav_b")
    args = ap.parse_args()

    ckpt = torch.load(WORK / "ckpt" / "prosody_encoder.pt", map_location="cpu")
    enc = ProsodyEncoder()
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    from speechbrain.inference.speaker import EncoderClassifier
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(WORK / "pretrained" / "ecapa"),
        run_opts={"device": "cpu"})

    pa, pb = prosody_emb(enc, args.wav_a), prosody_emb(enc, args.wav_b)
    ea, eb = ecapa_emb(clf, args.wav_a), ecapa_emb(clf, args.wav_b)
    s_pros = float(pa @ pb)
    s_ecapa = float(ea @ eb)

    print("\n=== GhostPrint: same person behind these two calls? ===")
    print(f"  timbre voiceprint (ECAPA) cosine : {s_ecapa:+.3f}")
    print(f"  prosody fingerprint cosine       : {s_pros:+.3f}")
    if s_ecapa < 0.35 and s_pros > 0.55:
        print("  verdict: VOICE CHANGED, SPEAKER LIKELY SAME "
              "-> possible disguised repeat fraudster")
    elif s_ecapa > 0.35 and s_pros > 0.4:
        print("  verdict: likely same speaker")
    elif s_ecapa < 0.2 and s_pros < 0.3:
        print("  verdict: likely different speakers")
    else:
        print("  verdict: inconclusive — route to analyst")


if __name__ == "__main__":
    main()
