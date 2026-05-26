"""Train a non-autoregressive trajectory surrogate.

Surrogate: (per-step features) -> (lat_accel trajectory)
  - per-step features:  [action_t, roll_t, v_ego_t, a_ego_t, target_t]
  - output:             lat_accel[t] for t in cost window [100, 500)

Architecture: causal 1D dilated convolution stack (TCN). Each output
position depends only on past + current inputs (no future leak). The
dilations let the receptive field cover the full 580-step segment.

Loss: MSE on lat_accel trajectory within [100, 500) plus an auxiliary
direct cost loss (so the surrogate focuses on the regions that drive
the eval metric).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Mirror the constants
CONTROL_START_IDX = 100
COST_END_IDX = 500
ACTION_HORIZON = COST_END_IDX - CONTROL_START_IDX
TRAJ_LEN = 580
DEL_T = 0.1
LAT_COST_MULT = 50.0
N_FEATURES = 5  # action, roll, v_ego, a_ego, target


class TrajData(Dataset):
    def __init__(self, npz_path, normalise=True):
        d = np.load(npz_path, allow_pickle=True)
        self.actions = d["actions"].astype(np.float32)        # (N, 400)
        self.lat_traj = d["lat_traj"].astype(np.float32)      # (N, TRAJ_LEN)
        self.states = d["states"].astype(np.float32)          # (S, TRAJ_LEN, 4)
        self.seg_idx = d["seg_idx"].astype(np.int64)          # (N,)
        self.pre_steer = d["pre_steer"].astype(np.float32)    # (S, 100)
        self.costs = d["costs"].astype(np.float32)            # (N,)
        # Normalisation stats
        if normalise:
            self.state_mean = self.states.reshape(-1, 4).mean(0)
            self.state_std = self.states.reshape(-1, 4).std(0) + 1e-6
        else:
            self.state_mean = np.zeros(4, dtype=np.float32)
            self.state_std = np.ones(4, dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        sidx = self.seg_idx[idx]
        # Per-step feature matrix (TRAJ_LEN, 5): [action, roll, vego, aego, target]
        feats = np.zeros((TRAJ_LEN, N_FEATURES), dtype=np.float32)
        # Pre-control actions: dataset steer commands
        feats[:CONTROL_START_IDX, 0] = self.pre_steer[sidx]
        feats[CONTROL_START_IDX:CONTROL_START_IDX + ACTION_HORIZON, 0] = self.actions[idx]
        # State features (normalised)
        s = self.states[sidx]  # (TRAJ_LEN, 4)
        s_norm = (s - self.state_mean) / self.state_std
        feats[:, 1:] = s_norm
        # Targets: lat_accel within cost window [100, 500)
        target_traj = self.lat_traj[idx, CONTROL_START_IDX:COST_END_IDX]
        # Also pre-control "true" lat_accel (which is target for those steps)
        pre_lat = self.lat_traj[idx, :CONTROL_START_IDX]
        return (
            torch.from_numpy(feats),                  # (TRAJ_LEN, 5)
            torch.from_numpy(target_traj),            # (400,)
            torch.from_numpy(pre_lat),                # (100,), context conditioning
            torch.tensor(self.costs[idx], dtype=torch.float32),
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel, dilation=dilation)

    def forward(self, x):
        # x: (B, C, L)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TrajSurrogate(nn.Module):
    """Receives features (B, TRAJ_LEN, 5), outputs lat_accel (B, ACTION_HORIZON)."""

    def __init__(self, hidden=64, n_blocks=7, kernel=3):
        super().__init__()
        self.input_proj = nn.Conv1d(N_FEATURES, hidden, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dil = 2 ** i  # exponentially growing receptive field
            self.blocks.append(nn.Sequential(
                CausalConv1d(hidden, hidden, kernel, dil),
                nn.GELU(),
                CausalConv1d(hidden, hidden, kernel, dil),
            ))
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(n_blocks)])
        self.out_proj = nn.Conv1d(hidden, 1, 1)

    def forward(self, feats):
        # feats: (B, TRAJ_LEN, 5) -> (B, 5, TRAJ_LEN)
        x = feats.transpose(1, 2)
        x = self.input_proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            x = norm(x + blk(x))
        out = self.out_proj(x).squeeze(1)  # (B, TRAJ_LEN)
        return out[:, CONTROL_START_IDX:COST_END_IDX]  # (B, ACTION_HORIZON)


def compute_cost(pred, target):
    """Replicates real sim cost: lat * 50 + jerk."""
    lat_cost = ((pred - target) ** 2).mean(dim=-1) * 100.0
    jerk = (pred[:, 1:] - pred[:, :-1]) / DEL_T
    jerk_cost = (jerk ** 2).mean(dim=-1) * 100.0
    return LAT_COST_MULT * lat_cost + jerk_cost


def train(npz_path, out_path, epochs=80, batch_size=32, lr=2e-3, val_frac=0.1,
          hidden=64, n_blocks=7, sample_weight_low_cost=True):
    ds = TrajData(npz_path)
    # Sample-weighted training: emphasise low-cost (near-optimal) samples
    # where the optimizer actually lives.
    weights = np.ones(len(ds), dtype=np.float32)
    if sample_weight_low_cost:
        # weight proportional to 1 / (1 + cost/100) so low-cost samples get
        # ~5x higher weight than median-cost samples.
        weights = 1.0 / (1.0 + ds.costs / 100.0)
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ds))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_weights = weights[train_idx]
    train_weights = train_weights / train_weights.sum() * len(train_weights)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(train_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n_train={n_train}  n_val={n_val}")
    model = TrajSurrogate(hidden=hidden, n_blocks=n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"surrogate params: {n_params/1e3:.1f}k")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_loss = float("inf")
    for ep in range(epochs):
        # Train
        model.train()
        train_traj_loss = 0.0
        train_cost_loss = 0.0
        n = 0
        for feats, target_traj, pre_lat, cost_true in train_dl:
            feats = feats.to(device)
            target_traj = target_traj.to(device)
            cost_true = cost_true.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(feats)
            traj_loss = F.mse_loss(pred, target_traj)
            cost_pred = compute_cost(pred, target_traj.detach())  # cost using PRED as if it were lat_traj
            # Indirect: use predicted lat_accel to compute the same cost the real sim computes
            cost_pred_full = compute_cost(pred, feats[:, CONTROL_START_IDX:COST_END_IDX, 4])  # target feature is col 4
            cost_loss = F.mse_loss(cost_pred_full, cost_true) / 1000.0
            loss = traj_loss + 0.5 * cost_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_traj_loss += traj_loss.item() * feats.size(0)
            train_cost_loss += cost_loss.item() * feats.size(0)
            n += feats.size(0)
        sched.step()
        train_traj_loss /= n
        train_cost_loss /= n

        # Val
        model.eval()
        v_traj_loss = 0.0
        v_n = 0
        v_cost_err = 0.0
        with torch.no_grad():
            for feats, target_traj, pre_lat, cost_true in val_dl:
                feats = feats.to(device); target_traj = target_traj.to(device); cost_true = cost_true.to(device)
                pred = model(feats)
                v_traj_loss += F.mse_loss(pred, target_traj).item() * feats.size(0)
                cost_pred_full = compute_cost(pred, feats[:, CONTROL_START_IDX:COST_END_IDX, 4])
                v_cost_err += (cost_pred_full - cost_true).abs().mean().item() * feats.size(0)
                v_n += feats.size(0)
        v_traj_loss /= v_n
        v_cost_err /= v_n

        if v_traj_loss < best_val_loss:
            best_val_loss = v_traj_loss
            torch.save({
                "state_dict": model.state_dict(),
                "state_mean": ds.state_mean,
                "state_std": ds.state_std,
                "hidden": hidden,
                "n_blocks": n_blocks,
            }, out_path)
            tag = "  *"
        else:
            tag = ""
        print(f"ep={ep:03d}  train_traj={train_traj_loss:.5f}  train_cost={train_cost_loss:.5f}  "
              f"val_traj={v_traj_loss:.5f}  val_cost_abs_err={v_cost_err:.2f}{tag}")

    print(f"\nBest val_traj_loss: {best_val_loss:.5f}")
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="surrogate_data.npz")
    parser.add_argument("--out", default="surrogate.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=7)
    args = parser.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, hidden=args.hidden, n_blocks=args.n_blocks)


if __name__ == "__main__":
    main()
