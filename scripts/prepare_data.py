"""Step 1: select utterances and generate disguised copies.

Outputs under work/:
  manifests/train_clean.jsonl        dev-clean, 30 utts x 40 speakers
  manifests/train_disguised.jsonl    2 random train presets per train utt
  manifests/eval_enroll.jsonl        test-clean, 5 clean utts/speaker (watchlist)
  manifests/eval_probe_clean.jsonl   12 clean probes/speaker
  manifests/eval_probe_<preset>.jsonl  same probes, disguised (all 5 presets,
                                       incl. 'hard' which is UNSEEN in training)
  wav/...                            generated disguised audio
"""

import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from ghostprint.config import DISGUISE_PRESETS, TRAIN_PRESETS, WORK, data_cfg
from ghostprint.data import index_split, select_per_speaker, save_manifest, utt_row
from ghostprint.disguise import apply_disguise

MAN = WORK / "manifests"
WAV = WORK / "wav"


def _disguise_job(job):
    in_path, out_path, preset = job
    if not Path(out_path).exists():
        apply_disguise(in_path, out_path, preset)
    return out_path


def main():
    rng = random.Random(data_cfg.seed)
    MAN.mkdir(parents=True, exist_ok=True)

    # ---------------- train side (dev-clean) ----------------
    train_idx = index_split(data_cfg.train_split)
    train_sel = select_per_speaker(train_idx, data_cfg.train_utts_per_spk, data_cfg.seed)
    print(f"train speakers: {len(train_sel)}")

    train_clean, train_disg, jobs = [], [], []
    for spk, utts in train_sel.items():
        for u in utts:
            train_clean.append(utt_row(u, condition="clean"))
            for preset in rng.sample(TRAIN_PRESETS, 2):
                out = WAV / "train" / preset / f"{u.utt_id}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((u.path, str(out), preset))
                train_disg.append(utt_row(u, condition=preset, path=str(out)))
    save_manifest(train_clean, MAN / "train_clean.jsonl")
    save_manifest(train_disg, MAN / "train_disguised.jsonl")

    # ---------------- eval side (test-clean, unseen speakers) ----------------
    need = data_cfg.enroll_utts_per_spk + data_cfg.probe_utts_per_spk
    eval_idx = index_split(data_cfg.eval_split)
    eval_sel = select_per_speaker(eval_idx, need, data_cfg.seed + 1)
    print(f"eval speakers: {len(eval_sel)}")

    enroll, probe_clean = [], []
    probe_disg = {p: [] for p in DISGUISE_PRESETS}
    for spk, utts in eval_sel.items():
        for u in utts[: data_cfg.enroll_utts_per_spk]:
            enroll.append(utt_row(u, condition="clean"))
        for u in utts[data_cfg.enroll_utts_per_spk:]:
            probe_clean.append(utt_row(u, condition="clean"))
            for preset in DISGUISE_PRESETS:
                out = WAV / "eval" / preset / f"{u.utt_id}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((u.path, str(out), preset))
                probe_disg[preset].append(utt_row(u, condition=preset, path=str(out)))
    save_manifest(enroll, MAN / "eval_enroll.jsonl")
    save_manifest(probe_clean, MAN / "eval_probe_clean.jsonl")
    for preset, rows in probe_disg.items():
        save_manifest(rows, MAN / f"eval_probe_{preset}.jsonl")

    # ---------------- run disguise synthesis ----------------
    print(f"synthesizing {len(jobs)} disguised wavs ...")
    with Pool(4) as pool:
        list(tqdm(pool.imap_unordered(_disguise_job, jobs, chunksize=8),
                  total=len(jobs)))
    print("prepare_data: done")


if __name__ == "__main__":
    main()
