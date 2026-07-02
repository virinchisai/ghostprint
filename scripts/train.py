"""Step 3: train the prosody encoder with disguise augmentation.

Each item = one (speaker, utterance); at sample time we randomly pick the
clean version or one of its disguised variants, so the encoder is pushed
to map a speaker's clean and disguised speech to the same point =
disguise/clone invariance. The 'hard' preset is never seen in training.
"""

import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ghostprint.config import WORK, train_cfg, data_cfg
from ghostprint.data import load_manifest
from ghostprint.model import ProsodyEncoder, AAMSoftmax, collate

MAN = WORK / "manifests"
PROS = WORK / "feat" / "prosody"
CKPT = WORK / "ckpt"


class ProsodySet(Dataset):
    def __init__(self):
        with open(PROS / "train_clean.pkl", "rb") as f:
            clean = pickle.load(f)
        with open(PROS / "train_disguised.pkl", "rb") as f:
            disg = pickle.load(f)
        rows = load_manifest(MAN / "train_clean.jsonl")
        self.speakers = sorted({r["speaker"] for r in rows})
        self.spk2label = {s: i for i, s in enumerate(self.speakers)}

        # utt_id -> list of feature dicts (clean first, then variants)
        self.variants: dict[str, list] = {}
        self.items = []
        for r in rows:
            uid = r["utt_id"]
            self.variants[uid] = [clean[f"{uid}|clean"]]
            self.items.append((uid, self.spk2label[r["speaker"]]))
        for key, feats in disg.items():
            uid = key.split("|")[0]
            if uid in self.variants:
                self.variants[uid].append(feats)
        self.rng = random.Random(data_cfg.seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        uid, label = self.items[i]
        vars_ = self.variants[uid]
        if len(vars_) > 1 and self.rng.random() < train_cfg.disguise_prob:
            feats = self.rng.choice(vars_[1:])
        else:
            feats = vars_[0]
        frames = feats["frames"]
        # light augmentation: random crop to <= 12s, small feature noise
        max_t = 12 * 50
        if frames.shape[0] > max_t:
            s = self.rng.randrange(frames.shape[0] - max_t)
            frames = frames[s:s + max_t]
        frames = frames + np.random.randn(*frames.shape).astype(np.float32) * 0.02
        return {"frames": frames, "stats": feats["stats"], "label": label}


def main():
    torch.manual_seed(data_cfg.seed)
    ds = ProsodySet()
    dl = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)
    enc = ProsodyEncoder(train_cfg.frame_dim, train_cfg.stats_dim,
                         train_cfg.hidden, train_cfg.emb_dim)
    head = AAMSoftmax(train_cfg.emb_dim, len(ds.speakers),
                      train_cfg.aam_margin, train_cfg.aam_scale)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=train_cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_cfg.epochs)

    CKPT.mkdir(parents=True, exist_ok=True)
    for epoch in range(train_cfg.epochs):
        enc.train()
        tot, nb, correct, n = 0.0, 0, 0, 0
        for frames, lengths, stats, labels in dl:
            emb = enc(frames, lengths, stats)
            loss = head(emb, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
            with torch.no_grad():
                logits = emb @ torch.nn.functional.normalize(head.weight, dim=-1).T
                correct += (logits.argmax(-1) == labels).sum().item()
                n += len(labels)
        sched.step()
        print(f"epoch {epoch+1:02d}/{train_cfg.epochs} "
              f"loss={tot/nb:.3f} train_acc={correct/n:.3f}")

    torch.save({"encoder": enc.state_dict(),
                "speakers": ds.speakers,
                "cfg": vars(train_cfg)}, CKPT / "prosody_encoder.pt")
    print(f"saved {CKPT / 'prosody_encoder.pt'}")


if __name__ == "__main__":
    main()
