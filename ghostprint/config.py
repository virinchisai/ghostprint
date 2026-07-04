"""Central configuration for data prep, training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"          # generated wavs, features, checkpoints
RESULTS = ROOT / "results"


@dataclass
class DataConfig:
    librispeech_root: Path = DATA / "LibriSpeech"
    train_split: str = "dev-clean"    # 40 speakers -> encoder training
    eval_split: str = "test-clean"    # 40 unseen speakers -> open-set eval
    min_dur_s: float = 4.0
    max_dur_s: float = 20.0
    train_utts_per_spk: int = 30
    enroll_utts_per_spk: int = 5      # clean enrollment ("watchlist voiceprint")
    probe_utts_per_spk: int = 12      # probes, tested clean AND disguised
    sample_rate: int = 16000
    seed: int = 1337


@dataclass
class TrainConfig:
    frame_dim: int = 4                # zlogf0, voiced, zenergy, dzlogf0
    stats_dim: int = 12
    hidden: int = 128
    emb_dim: int = 192
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    # each training sample is drawn clean or disguised with this prob
    disguise_prob: float = 0.6
    device: str = "cpu"


# disguise presets used to SIMULATE a fraudster hiding their voice.
# name -> (semitone shift, formant ratio, duration factor)
DISGUISE_PRESETS: dict[str, tuple[float, float, float]] = {
    "pitch_up":    (+3.0, 1.00, 1.00),
    "pitch_down":  (-3.0, 1.00, 1.00),
    "gender_up":   (+4.0, 1.14, 1.00),
    "gender_down": (-4.0, 0.88, 1.00),
    "hard":        (-2.0, 1.10, 0.94),   # + mild tempo change: stress test
    # formant dose-response sweep (pitch held ~constant, formants pushed):
    "fsweep_096":  (-1.0, 0.96, 1.00),
    "fsweep_088":  (-1.0, 0.88, 1.00),
    "fsweep_080":  (-1.0, 0.80, 1.00),
    "fsweep_072":  (-1.0, 0.72, 1.00),
    "fsweep_065":  (-1.0, 0.65, 1.00),
}

# formant ratios used by scripts/sweep.py, in order
SWEEP_PRESETS = ["fsweep_096", "fsweep_088", "fsweep_080", "fsweep_072", "fsweep_065"]

# presets seen during training augmentation (held-out preset tests generalization)
TRAIN_PRESETS = ["pitch_up", "pitch_down", "gender_up", "gender_down"]

data_cfg = DataConfig()
train_cfg = TrainConfig()
