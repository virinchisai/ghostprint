"""Behavioral channel proof-of-concept: re-identify "fraudsters" from
simulated IVR interaction patterns alone (no audio at all).

Each synthetic fraudster has stable habits: preferred menu paths
(Markov transition matrix), inter-key timing (log-normal), and an
input-error rate. We sample independent sessions and test whether
sessions link back to the right fraudster — the behavioral analogue
of a voiceprint, untouched by any voice disguise.

This channel is SIMULATED (clearly labeled as such) — it demonstrates
that the fusion architecture extends beyond audio, matching how real
IVR keypress/menu telemetry would plug in.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ghostprint.config import RESULTS

N_FRAUDSTERS = 60
N_MENUS = 8
SESSIONS_ENROLL = 4
SESSIONS_PROBE = 3
EVENTS = (15, 40)
RNG = np.random.default_rng(1337)


def make_profile():
    trans = RNG.dirichlet(np.full(N_MENUS, 0.35), size=N_MENUS)  # spiky rows
    delay_mu = RNG.uniform(-0.9, 0.5)        # log-seconds between keys
    delay_sigma = RNG.uniform(0.25, 0.6)
    err = RNG.uniform(0.0, 0.12)
    return trans, delay_mu, delay_sigma, err


def sample_session(profile):
    trans, mu, sigma, err = profile
    n = RNG.integers(*EVENTS)
    state = RNG.integers(N_MENUS)
    path, delays, errors = [state], [], 0
    for _ in range(n - 1):
        state = RNG.choice(N_MENUS, p=trans[state])
        path.append(state)
        delays.append(RNG.lognormal(mu, sigma))
        errors += RNG.random() < err
    return path, np.array(delays), errors / n


def session_features(path, delays, err_rate):
    bigram = np.zeros((N_MENUS, N_MENUS))
    for a, b in zip(path[:-1], path[1:]):
        bigram[a, b] += 1
    bigram = bigram.flatten() / max(len(path) - 1, 1)
    timing = [delays.mean(), delays.std(),
              np.percentile(delays, 10), np.percentile(delays, 90)]
    f = np.concatenate([bigram, timing, [err_rate]])
    return f / (np.linalg.norm(f) + 1e-9)


def main():
    profiles = [make_profile() for _ in range(N_FRAUDSTERS)]
    centroids, probes, labels = [], [], []
    for i, p in enumerate(profiles):
        enr = [session_features(*sample_session(p)) for _ in range(SESSIONS_ENROLL)]
        c = np.mean(enr, axis=0)
        centroids.append(c / np.linalg.norm(c))
        for _ in range(SESSIONS_PROBE):
            probes.append(session_features(*sample_session(p)))
            labels.append(i)
    C = np.stack(centroids)
    S = np.stack(probes) @ C.T
    labels = np.array(labels)

    rank1 = float((S.argmax(1) == labels).mean() * 100)
    tgt = S[np.arange(len(labels)), labels]
    mask = np.ones_like(S, bool)
    mask[np.arange(len(labels)), labels] = False
    non = S[mask]
    thr = np.sort(np.concatenate([tgt, non]))
    fnr = np.array([(tgt < t).mean() for t in thr])
    fpr = np.array([(non >= t).mean() for t in thr])
    i = np.argmin(np.abs(fnr - fpr))
    eer = float((fnr[i] + fpr[i]) / 2 * 100)

    out = {"n_fraudsters": N_FRAUDSTERS, "rank1": round(rank1, 2),
           "eer": round(eer, 2), "note": "simulated IVR behavior channel"}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "ivr_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
