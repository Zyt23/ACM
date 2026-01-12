# -*- coding: utf-8 -*-
"""
Window dataset built on top of your FlightDataset_acm(seghead keep_len).

- Avoids materializing all windows: keeps an index map (seg_idx, start_step).
- Supports anomaly mixing using mechanism-inspired synthesis.
- IMPORTANT: supports tail-based split to avoid leakage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from data_provider.acm_anomaly_synth import AnomType, apply_anomaly, ACMColIndex


@dataclass
class WindowMeta:
    tail: str
    seg_start_time: str
    seg_index: int
    step_offset: int


class SegHeadWindowDataset(Dataset):
    """
    base_ds: your FlightDataset_acm
      base_ds.data[i] : [keep_len, 6]
      base_ds.window_tails[i], base_ds.window_start_times[i]
    """
    def __init__(self, base_ds, win_len: int = 96, stride: int = 96):
        super().__init__()
        self.base = base_ds
        self.win_len = int(win_len)
        self.stride = int(stride)

        if len(self.base) == 0:
            self._index: List[Tuple[int, int]] = []
            self._tails: List[str] = []
            self._times: List[str] = []
            return

        keep_len = int(self.base.data.shape[1])
        if keep_len < self.win_len:
            raise ValueError(f"keep_len={keep_len} < win_len={self.win_len}")

        n_sub = 1 + (keep_len - self.win_len) // self.stride

        idx_list: List[Tuple[int, int]] = []
        tails: List[str] = []
        times: List[str] = []

        for seg_idx in range(len(self.base)):
            tail = str(self.base.window_tails[seg_idx]) if hasattr(self.base, "window_tails") else "UNKNOWN"
            stime = str(self.base.window_start_times[seg_idx]) if hasattr(self.base, "window_start_times") else "UNKNOWN"
            for sub_id in range(n_sub):
                st = sub_id * self.stride
                idx_list.append((seg_idx, st))
                tails.append(tail)
                times.append(stime)

        self._index = idx_list
        self._tails = tails
        self._times = times

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i: int):
        seg_idx, st = self._index[i]
        seg = self.base[seg_idx]  # [keep_len,6]
        win = seg[st:st + self.win_len, :].astype(np.float32)  # [L,6]
        meta = WindowMeta(
            tail=self._tails[i],
            seg_start_time=self._times[i],
            seg_index=int(seg_idx),
            step_offset=int(st),
        )
        return win, meta

    @property
    def sample_tails(self) -> List[str]:
        return self._tails


class AnomalyMixDataset(Dataset):
    """
    Wrap a SegHeadWindowDataset:
      - with probability p_anom -> synth anomaly and label=kind
      - else label=NORMAL
    """
    def __init__(
        self,
        win_ds: SegHeadWindowDataset,
        p_anom: float = 0.5,
        anom_kinds: Optional[List[AnomType]] = None,
        seed: int = 42,
        synth_params: Optional[Dict] = None
    ):
        super().__init__()
        self.ds = win_ds
        self.p_anom = float(p_anom)
        self.kinds = anom_kinds or [
            AnomType.ENERGY_RATIO,
            AnomType.MISALIGN_TEMP,
            AnomType.MISALIGN_VALVE,
            AnomType.DRIFT,
            AnomType.OSCILLATION,
            AnomType.STUCK_VALVE,
            AnomType.NOISE,
        ]
        self.seed = int(seed)
        self.params = synth_params or {}
        self.idx = ACMColIndex()

        # label space: 0..K, where 0 is NORMAL, others are AnomType enum values
        # We'll train multi-class with num_classes = max(enum)+1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: int):
        win, meta = self.ds[i]
        rng = np.random.default_rng(self.seed + i)

        if rng.random() < self.p_anom:
            kind = self.kinds[int(rng.integers(0, len(self.kinds)))]
            win2, info = apply_anomaly(win, kind=kind, rng=rng, idx=self.idx, params=self.params)
            label = int(kind)
            info.update({"tail": meta.tail, "seg_start_time": meta.seg_start_time, "seg_index": meta.seg_index, "step_offset": meta.step_offset})
        else:
            win2 = win
            label = int(AnomType.NORMAL)
            info = {"kind": int(AnomType.NORMAL), "tail": meta.tail, "seg_start_time": meta.seg_start_time,
                    "seg_index": meta.seg_index, "step_offset": meta.step_offset}

        # return torch tensors for convenience
        x = torch.from_numpy(win2).float()          # [L,6]
        y = torch.tensor(label, dtype=torch.long)   # []
        return x, y, info


def split_indices_by_tail(
    tails: List[str],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Tail-level split: all windows from a tail go either train or val.
    """
    uniq = sorted(set([str(t) for t in tails]))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)

    n_val_tail = max(1, int(len(uniq) * float(val_ratio)))
    val_tails = set(uniq[:n_val_tail])

    tr_idx, va_idx = [], []
    for i, t in enumerate(tails):
        if str(t) in val_tails:
            va_idx.append(i)
        else:
            tr_idx.append(i)
    return tr_idx, va_idx
