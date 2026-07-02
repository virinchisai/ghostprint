"""LibriSpeech indexing and manifest building."""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import soundfile as sf

from .config import data_cfg


@dataclass
class Utt:
    utt_id: str
    speaker: str
    path: str
    dur: float


def index_split(split: str) -> list[Utt]:
    """Walk a LibriSpeech split and keep utterances in the duration band."""
    root = data_cfg.librispeech_root / split
    utts = []
    for flac in sorted(root.rglob("*.flac")):
        info = sf.info(str(flac))
        dur = info.frames / info.samplerate
        if data_cfg.min_dur_s <= dur <= data_cfg.max_dur_s:
            spk = flac.name.split("-")[0]
            utts.append(Utt(flac.stem, spk, str(flac), round(dur, 2)))
    return utts


def select_per_speaker(utts: list[Utt], k: int, seed: int) -> dict[str, list[Utt]]:
    rng = random.Random(seed)
    by_spk: dict[str, list[Utt]] = {}
    for u in utts:
        by_spk.setdefault(u.speaker, []).append(u)
    out = {}
    for spk, lst in sorted(by_spk.items()):
        lst = sorted(lst, key=lambda u: u.utt_id)
        rng.shuffle(lst)
        if len(lst) >= k:
            out[spk] = lst[:k]
    return out


def save_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def utt_row(u: Utt, **extra) -> dict:
    d = asdict(u)
    d.update(extra)
    return d
