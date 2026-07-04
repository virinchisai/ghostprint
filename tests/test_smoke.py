"""Fast smoke tests — exercise the core GhostPrint code paths with no data
download and no pretrained models. Runs in seconds; suitable for CI.

Covered: prosody feature extraction, disguise synthesis, the encoder forward
pass (shape + L2-normalization), the AAM-softmax head, and the collate fn.
The ECAPA baseline is intentionally out of scope here (heavy model download).
"""

import os

import numpy as np
import soundfile as sf
import torch

from ghostprint.features import extract
from ghostprint.disguise import apply_disguise
from ghostprint.model import ProsodyEncoder, AAMSoftmax, collate


def _make_wav(path, sr=16000, dur=4.0):
    """A voiced, harmonic-rich signal with amplitude modulation and two pauses
    so pitch / energy / pause features are all non-trivial."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0 = 120.0
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    env[int(sr * 1.0):int(sr * 1.3)] = 0.0   # pause 1
    env[int(sr * 2.5):int(sr * 2.9)] = 0.0   # pause 2
    sig = (sig * env).astype(np.float32)
    sig /= np.max(np.abs(sig)) + 1e-9
    sf.write(path, sig * 0.8, sr)


def test_feature_shapes(tmp_path):
    wav = str(tmp_path / "a.wav"); _make_wav(wav)
    f = extract(wav)
    assert f["frames"].ndim == 2 and f["frames"].shape[1] == 4
    assert f["stats"].shape == (12,)
    assert np.isfinite(f["frames"]).all()
    assert np.isfinite(f["stats"]).all()


def test_disguise_runs(tmp_path):
    wav = str(tmp_path / "a.wav"); _make_wav(wav)
    out = str(tmp_path / "d.wav")
    apply_disguise(wav, out, "gender_down")
    assert os.path.exists(out)
    g = extract(out)
    assert g["frames"].shape[1] == 4


def test_encoder_forward_and_norm():
    enc = ProsodyEncoder().eval()
    B, T = 3, 200
    frames = torch.randn(B, T, 4)
    lengths = torch.tensor([T, T - 10, T - 20])
    stats = torch.randn(B, 12)
    with torch.no_grad():
        emb = enc(frames, lengths, stats)
    assert emb.shape == (B, 192)
    assert torch.allclose(emb.norm(dim=1), torch.ones(B), atol=1e-4)


def test_aam_softmax_scalar_loss():
    head = AAMSoftmax(192, 10)
    emb = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
    loss = head(emb, torch.tensor([0, 1, 2, 3]))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_collate_batches():
    batch = [
        {"frames": np.random.randn(50, 4).astype(np.float32),
         "stats": np.random.randn(12).astype(np.float32), "label": i}
        for i in range(3)
    ]
    frames, lengths, stats, labels = collate(batch)
    assert frames.shape[0] == 3
    assert stats.shape == (3, 12)
    assert labels.shape == (3,)
