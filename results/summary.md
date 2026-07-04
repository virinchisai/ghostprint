# GhostPrint evaluation — 40 unseen speakers (LibriSpeech test-clean)

Watchlist protocol: 5-utt clean enrollment, 12 probes/spk.
'hard' disguise preset was never seen in training.

## EER % (lower better)

| condition | ECAPA (voiceprint) | GhostPrint (prosody) | Fusion |
|---|---|---|---|
| clean | 0.0 | 19.81 | 0.21 |
| pitch_up | 0.0 | 21.25 | 0.26 |
| pitch_down | 0.01 | 21.01 | 0.42 |
| gender_up | 3.33 | 21.46 | 4.54 |
| gender_down | 5.25 | 21.23 | 5.21 |
| hard | 1.22 | 22.91 | 2.08 |

## Rank-1 identification % (higher better)

| condition | ECAPA (voiceprint) | GhostPrint (prosody) | Fusion |
|---|---|---|---|
| clean | 100.0 | 29.79 | 100.0 |
| pitch_up | 100.0 | 29.79 | 100.0 |
| pitch_down | 100.0 | 27.5 | 99.38 |
| gender_up | 83.54 | 29.38 | 87.5 |
| gender_down | 74.17 | 25.42 | 84.17 |
| hard | 97.29 | 26.88 | 96.25 |

## Watchlist hit-rate % @ 1% FAR (higher better)

| condition | ECAPA (voiceprint) | GhostPrint (prosody) | Fusion |
|---|---|---|---|
| clean | 100.0 | 20.83 | 100.0 |
| pitch_up | 100.0 | 20.21 | 100.0 |
| pitch_down | 100.0 | 17.92 | 100.0 |
| gender_up | 88.12 | 20.83 | 90.0 |
| gender_down | 79.17 | 17.08 | 84.58 |
| hard | 98.54 | 18.12 | 96.88 |
