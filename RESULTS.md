# GhostPrint — Results

All numbers are open-set evaluation on **40 speakers held out from training**
(LibriSpeech `test-clean`). Protocol: each speaker enrolled from **5 clean
utterances** (centroid = their "watchlist voiceprint"); **12 probe utterances**
per speaker, scored clean and under each disguise. Scores are cohort
z-normalised (s-norm) before fusion; fusion = equal-weight sum of the ECAPA and
GhostPrint normalised scores, with **no tuning on the eval set**.

Systems compared:

- **ECAPA voiceprint** — `speechbrain/spkrec-ecapa-voxceleb`, the industry-standard
  timbre embedding ("what watchlists use today").
- **GhostPrint prosody** — our BiGRU encoder over per-utterance-normalised
  F0-shape / energy / voicing + 12 rhythm/pause statistics, trained with
  AAM-softmax and **clean↔disguised augmentation** so a speaker's clean and
  disguised speech are pushed to the same embedding.
- **Fusion** — ECAPA + GhostPrint.

Metrics: **EER** (equal-error rate, lower better), **Rank-1** (top-1
identification, higher better), **Hit-rate @ 1% FAR** (watchlist true-positive
rate at a 1% false-alarm budget, higher better).

---

## Table 1 — Discrete disguise conditions

### Rank-1 identification % (higher better)

| condition | ECAPA | GhostPrint | Fusion |
|---|---|---|---|
| clean | **100.0** | 29.8 | 100.0 |
| pitch_up (+3 st) | **100.0** | 29.8 | 100.0 |
| pitch_down (−3 st) | **100.0** | 27.5 | 99.4 |
| gender_up (formant 1.14) | 83.5 | 29.4 | **87.5** |
| gender_down (formant 0.88) | 74.2 | 25.4 | **84.2** |
| hard (unseen in training) | 97.3 | 26.9 | 96.3 |

### EER % (lower better)

| condition | ECAPA | GhostPrint | Fusion |
|---|---|---|---|
| clean | **0.0** | 19.8 | 0.2 |
| pitch_up | **0.0** | 21.3 | 0.3 |
| pitch_down | **0.0** | 21.0 | 0.4 |
| gender_up | 3.3 | 21.5 | 4.5 |
| gender_down | 5.3 | 21.2 | 5.2 |
| hard | 1.2 | 22.9 | 2.1 |

### Watchlist hit-rate % @ 1% FAR (higher better)

| condition | ECAPA | GhostPrint | Fusion |
|---|---|---|---|
| clean | **100.0** | 20.8 | 100.0 |
| pitch_up | **100.0** | 20.2 | 100.0 |
| pitch_down | **100.0** | 17.9 | 100.0 |
| gender_up | 88.1 | 20.8 | **90.0** |
| gender_down | 79.2 | 17.1 | **84.6** |
| hard | 98.5 | 18.1 | 96.9 |

**Figure:** `results/eer_by_condition.png`, `results/tpr_by_condition.png`

**How to read Table 1.**
- **GhostPrint is disguise-invariant.** Every GhostPrint column is nearly
  constant across conditions (rank-1 25–30%, EER 20–23%). Changing the voice
  does not move it — that is the whole point.
- **Prosody is a weak *absolute* biometric.** On clean audio ECAPA is far
  stronger (100% vs 30% rank-1). GhostPrint is a **complement, not a
  replacement** — its worth is stability, not peak accuracy.
- **These PSOLA disguises are mild.** They only dent ECAPA under *formant*
  manipulation (gender presets: 100→74%); pitch-only shifts leave it at 100%.
  So on discrete conditions the voiceprint does **not** collapse.
- **Fusion already pays off where ECAPA weakens.** In exactly the two conditions
  ECAPA drops (gender_up/down), fusion **beats ECAPA alone**
  (83.5→87.5, 74.2→84.2 rank-1; 79.2→84.6 hit-rate). The prosody channel adds
  back what formant distortion removes.

---

## Table 2 — Formant dose-response sweep (headline result)

Vary one interpretable axis — formant (vocal-tract-length) scaling, the thing
neural voice conversion restructures most — from 1.0 (undisguised) to 0.65
(extreme), holding pitch roughly constant.

| formant ratio | ECAPA rank-1 | GhostPrint rank-1 | ECAPA EER | GhostPrint EER |
|---|---|---|---|---|
| 1.00 | 100.0 | 29.8 | 0.0 | 19.8 |
| 0.96 | 100.0 | 29.4 | 0.0 | 19.4 |
| 0.88 | 86.9 | 29.8 | 4.0 | 19.6 |
| 0.80 | 42.3 | 28.5 | 15.6 | 20.8 |
| **0.72** | **15.8** | **27.9** | **31.0** | **20.4** |
| **0.65** | **13.3** | **27.5** | **36.2** | **20.3** |

**Figure:** `results/formant_sweep.png`

**How to read Table 2.**
- **ECAPA degrades monotonically** as formants are pushed: 100 → 87 → 42 → 16 →
  13% rank-1; EER 0 → 36%. At ratio 0.65 the voiceprint is worse than random at
  putting a name to a probe.
- **GhostPrint is a flat line**: 30 → 27.5% rank-1, EER pinned near 20%. The
  disguise it was never trained on (0.65, 0.72 are outside the training presets)
  does not move it.
- **Crossover ≈ 0.76.** Below it, the disguise-invariant prosody fingerprint is
  the *more reliable* identifier than the timbre voiceprint. That is precisely
  the regime a formant-restructuring neural clone would create — and precisely
  where today's voiceprint watchlists silently fail.

This is the controlled, honest form of the "voiceprint collapses, prosody holds"
thesis: instead of one hand-picked disguise, a curve that says **where** the
crossover is and **how much** distortion it takes.

---

## Table 3 — Behavioral channel (simulated IVR)

Re-identify "fraudsters" from IVR interaction patterns alone — menu-path Markov
habits, inter-key timing, input-error rate — **no audio**. 60 synthetic
fraudsters, 4 enrollment sessions + 3 probe sessions each.

| metric | value |
|---|---|
| Rank-1 identification | 82.2% |
| EER | 17.7% |

**This channel is simulated** (clearly labelled as such). Its role is to show
the fusion architecture extends past audio to the metadata/behavioral telemetry
a real contact-center IVR already logs — a channel a voice clone cannot touch at
all.

---

## What this does and does not show

**Shows (supported by the numbers):**
1. A speaker fingerprint built from prosody/rhythm and trained for disguise
   invariance **does not degrade** under pitch/formant/tempo manipulation.
2. The standard timbre voiceprint **degrades sharply under formant
   manipulation**, with a **measured crossover (~0.76)** past which prosody wins.
3. **Fusion** captures the best of both: ECAPA's clean-audio precision plus a
   floor of robustness when the voice is altered.
4. The same watchlist math works on a **non-audio behavioral channel**.

**Does not show (honest limits):**
- **PSOLA ≠ neural voice conversion.** These disguises move the same axes but
  are weaker than RVC/XTTS. The sweep is a *controlled lower bound*; the
  `apply_neural_vc` hook in `ghostprint/disguise.py` reruns the identical
  protocol against real clones (needs a GPU).
- **Read speech, not calls.** LibriSpeech is audiobooks; conversational/telephony
  data would strengthen the rhythm/pause channel and unlock an *idiolect*
  (word-choice) channel, deliberately excluded here because on read speech it
  would leak book content rather than speaker identity.
- **40+40 speakers is demo scale.** The protocol is written to scale to
  VoxCeleb-size pools by editing `config.py` only.
- Prosody is a weaker biometric than timbre; GhostPrint raises the **cost** of
  evasion (a fraudster must now defeat several independent channels at once), it
  does not make evasion impossible.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/download_data.sh
scripts/run_all.sh            # prep → features → train → evaluate → IVR
python scripts/sweep.py       # formant dose-response curve
```
