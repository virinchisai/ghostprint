"""Prosody encoder + AAM-softmax training head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodyEncoder(nn.Module):
    """BiGRU over frame-level prosody + utterance stats -> 192-d embedding."""

    def __init__(self, frame_dim=4, stats_dim=12, hidden=128, emb_dim=192):
        super().__init__()
        self.gru = nn.GRU(frame_dim, hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.stats_mlp = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, 64), nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 4 + 64, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, emb_dim),
        )

    def forward(self, frames, lengths, stats):
        packed = nn.utils.rnn.pack_padded_sequence(
            frames, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1), device=out.device)[None, :]
                < lengths[:, None]).float().unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        out_masked = out.masked_fill(mask == 0, -1e4)
        mx = out_masked.max(1).values
        z = torch.cat([mean, mx, self.stats_mlp(stats)], dim=-1)
        emb = self.proj(z)
        return F.normalize(emb, dim=-1)


class AAMSoftmax(nn.Module):
    """Additive angular margin classification head (ArcFace-style)."""

    def __init__(self, emb_dim, n_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, emb, labels):
        cos = F.linear(emb, F.normalize(self.weight, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.margin)
        onehot = F.one_hot(labels, cos.size(1)).float()
        logits = self.scale * (onehot * target + (1 - onehot) * cos)
        return F.cross_entropy(logits, labels)


def collate(batch):
    frames = [torch.from_numpy(b["frames"]) for b in batch]
    lengths = torch.tensor([f.size(0) for f in frames])
    frames = nn.utils.rnn.pad_sequence(frames, batch_first=True)
    stats = torch.stack([torch.from_numpy(b["stats"]) for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    return frames, lengths, stats, labels
