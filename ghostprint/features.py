"""Clone-surviving prosodic features.

Everything here is deliberately *relative* (z-scored per utterance,
semitone-relative to the utterance's own median F0) so that shifting
absolute pitch or formants — what a disguise or clone changes — moves
the features as little as possible, while the speaker's intonation
shape, rhythm and pause habits remain.

Outputs per utterance:
  frames: (T, 4) float32 at 50 fps -> [zlog_f0, voiced, z_energy, d_zlog_f0]
  stats:  (12,)  float32 utterance-level rhythm/pause statistics
"""

from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import find_peaks

FPS = 50                # feature frames per second
HOP_S = 1.0 / FPS


def _load_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def extract(path: str | Path) -> dict[str, np.ndarray]:
    y, sr = _load_mono_16k(path)
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    dur = sound.get_total_duration()

    pitch = sound.to_pitch(time_step=HOP_S, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]              # 0 where unvoiced
    n = len(f0)
    times = pitch.xs()

    # frame energy on the same grid
    intensity = sound.to_intensity(minimum_pitch=60, time_step=HOP_S)
    energy = np.array(
        [intensity.get_value(t) if not np.isnan(intensity.get_value(t) or np.nan) else 0.0
         for t in times],
        dtype=np.float64,
    )
    energy = np.nan_to_num(energy, nan=0.0)

    voiced = (f0 > 0).astype(np.float32)
    logf0 = np.zeros(n)
    logf0[f0 > 0] = np.log2(f0[f0 > 0])

    # per-utterance normalization => invariance to absolute pitch level
    v = logf0[f0 > 0]
    if len(v) >= 5:
        mu, sd = v.mean(), max(v.std(), 1e-3)
    else:
        mu, sd = 0.0, 1.0
    zlogf0 = np.where(f0 > 0, (logf0 - mu) / sd, 0.0)
    dzlogf0 = np.gradient(zlogf0) * voiced

    e_mu, e_sd = energy.mean(), max(energy.std(), 1e-3)
    zenergy = (energy - e_mu) / e_sd

    frames = np.stack([zlogf0, voiced, zenergy, dzlogf0], axis=1).astype(np.float32)

    stats = _utterance_stats(f0, voiced, zenergy, dur)
    return {"frames": frames, "stats": stats}


def _utterance_stats(f0: np.ndarray, voiced: np.ndarray,
                     zenergy: np.ndarray, dur: float) -> np.ndarray:
    """12 rhythm/pause statistics, all pitch-level independent."""
    # pause structure from unvoiced runs (>= 200 ms)
    pauses = []
    run = 0
    for flag in voiced:
        if flag < 0.5:
            run += 1
        else:
            if run >= int(0.2 * FPS):
                pauses.append(run / FPS)
            run = 0
    if run >= int(0.2 * FPS):
        pauses.append(run / FPS)
    pauses = np.array(pauses) if pauses else np.array([0.0])

    # pseudo-syllable rate: energy peaks inside voiced regions
    env = zenergy * voiced
    peaks, _ = find_peaks(env, distance=int(0.1 * FPS), height=0.0)
    syl_rate = len(peaks) / max(dur, 1e-3)

    vf = f0[f0 > 0]
    if len(vf) >= 5:
        med = np.median(vf)
        st = 12.0 * np.log2(vf / med)          # semitones re. own median
        f0_range = float(np.percentile(st, 95) - np.percentile(st, 5))
        f0_std = float(st.std())
        d_st = np.diff(st)
        f0_slope = float(np.abs(d_st).mean()) if len(d_st) else 0.0
    else:
        f0_range = f0_std = f0_slope = 0.0

    # voiced-run durations ~ articulation habit
    vruns = []
    run = 0
    for flag in voiced:
        if flag > 0.5:
            run += 1
        else:
            if run > 0:
                vruns.append(run / FPS)
            run = 0
    if run > 0:
        vruns.append(run / FPS)
    vruns = np.array(vruns) if vruns else np.array([0.0])

    stats = np.array([
        len(pauses) / max(dur, 1e-3),      # pause rate
        pauses.mean(),                     # mean pause dur
        pauses.std(),                      # pause dur variability
        pauses.max(),                      # longest pause
        voiced.mean(),                     # voiced fraction
        syl_rate,                          # speaking-rate proxy
        vruns.mean(),                      # mean voiced-run
        vruns.std(),                       # voiced-run variability
        f0_range,                          # intonation range (semitones)
        f0_std,                            # intonation spread
        f0_slope,                          # pitch movement speed
        float(zenergy.std()),              # energy modulation
    ], dtype=np.float32)
    return np.nan_to_num(stats, nan=0.0)
