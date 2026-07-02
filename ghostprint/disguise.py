"""Voice-disguise simulation.

Emulates a fraudster masking their identity with pitch/formant/tempo
manipulation (the same axes neural voice conversion changes). Uses
Praat's PSOLA "Change gender" via parselmouth, which resynthesizes the
voice — enough to break a timbre-based speaker embedding while leaving
intonation *shape* and rhythm largely intact.

A hook for neural VC (e.g. RVC / FreeVC / kNN-VC) is provided so the
same pipeline can be rerun with real cloning systems on a GPU box.
"""

from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from .config import DISGUISE_PRESETS


def _median_f0(sound: parselmouth.Sound) -> float:
    pitch = sound.to_pitch(time_step=0.02, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return 150.0
    return float(np.median(f0))


def apply_disguise(in_path: str | Path, out_path: str | Path, preset: str) -> None:
    """Apply one named disguise preset and write a 16 kHz wav."""
    semitones, formant_ratio, duration_factor = DISGUISE_PRESETS[preset]
    sound = parselmouth.Sound(str(in_path))
    new_median = _median_f0(sound) * (2.0 ** (semitones / 12.0))
    disguised = call(
        sound, "Change gender",
        75, 600,                 # pitch floor / ceiling for analysis
        formant_ratio,
        new_median,
        1.0,                     # pitch range factor (keep intonation range)
        duration_factor,
    )
    disguised = call(disguised, "Resample", 16000, 50)
    disguised.save(str(out_path), "WAV")


def apply_neural_vc(in_path: str | Path, out_path: str | Path, target_voice: str) -> None:
    """Placeholder hook: swap in a neural voice-conversion system here.

    The evaluation protocol is unchanged — only this function needs to
    change to test against real cloning (RVC, FreeVC, kNN-VC, XTTS).
    """
    raise NotImplementedError(
        "Neural VC requires a GPU environment; see README 'Scaling up'."
    )
