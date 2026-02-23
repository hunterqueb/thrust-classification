import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility

from qutils.ml.classifer import prepareThrustClassificationDatasets
from qutils.ml.utils import getDevice, printModelParmSize
from qutils.tictoc import timer

import sys
from contextlib import redirect_stdout, redirect_stderr

CLASS_LABELS = ["No Thrust", "Chemical", "Electric", "Impulsive"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LSTM ensemble for thrust classification + temporal reachability rollout"
    )

    parser.add_argument("--systems", type=int, default=800, help="Number of systems")
    parser.add_argument("--propMin", type=int, default=30, help="Minimum propagation time [min]")
    parser.add_argument("--orbit", type=str, default="leo", help="Orbit type")
    parser.add_argument("--test", type=str, default=None, help="Test orbit type")
    parser.add_argument("--testSys", type=int, default=800, help="Test systems if --test differs")
    parser.add_argument("--OE", action="store_true", help="Use orbital elements")
    parser.add_argument("--noise", action="store_true", help="Enable noise in dataset prep")
    parser.add_argument("--velNoise", type=float, default=1e-3, help="Velocity noise std")
    parser.add_argument("--norm", action="store_true", help="Normalize semi-major axis")

    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping")

    parser.add_argument("--lookback", type=int, default=4, help="Lookback length for regression windows")
    parser.add_argument("--hidden", type=int, default=96, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=1, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout")
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Regression loss weight")

    parser.add_argument("--rollout_steps", type=int, default=40, help="Autoregressive rollout steps")
    parser.add_argument("--save_dir", type=str, default="plots/reachability_ensemble", help="Output directory")
    parser.add_argument("--model_dir", type=str, default="plots/reachability_ensemble/models", help="Model checkpoint directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to model checkpoint to load (overrides training)")

    return parser.parse_args()


@dataclass
class NormState:
    mu: torch.Tensor
    sig: torch.Tensor


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMThrustReachabilityEnsemble(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, output_size, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, output_size),
                )
                for _ in range(num_classes)
            ]
        )

        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x: (B, T, D)
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]

        logits = self.class_head(h_last)  # (B, C)
        expert_preds = torch.stack([head(h_last) for head in self.reg_heads], dim=1)  # (B, C, D)

        probs = torch.softmax(logits, dim=1)
        mixed_pred = torch.sum(expert_preds * probs.unsqueeze(-1), dim=1)  # (B, D)
        return logits, expert_preds, mixed_pred


class JointWindowDataset(data.Dataset):
    def __init__(self, x, y_cls, y_reg):
        self.x = x
        self.y_cls = y_cls
        self.y_reg = y_reg

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y_cls[idx], self.y_reg[idx]


class TrajectoryDataset(data.Dataset):
    def __init__(self, x, y_cls):
        self.x = x
        self.y_cls = y_cls

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y_cls[idx]


def infer_split_ratios(train_ratio: float):
    if abs(train_ratio - 0.7) < 1e-9:
        return 0.7, 0.15, 0.15
    val_ratio = train_ratio
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    if test_ratio <= 0:
        raise ValueError("train_ratio is too large for this script's split logic.")
    return train_ratio, val_ratio, test_ratio


def load_classification_trajectories(args):
    import yaml

    with open("data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    train_ratio, val_ratio, test_ratio = infer_split_ratios(args.train_ratio)

    yaml_cfg = {
        "useOE": args.OE,
        "useNorm": args.norm,
        "useNoise": args.noise,
        "useEnergy": False,
        "prop_time": args.propMin,
        "orbit": args.orbit,
        "systems": args.systems,
        "test_dataset": args.test if args.test is not None else args.orbit,
        "test_systems": args.testSys if args.test is not None else args.systems,
    }

    _, _, _, train_data, train_label, val_data, val_label, test_data, test_label = prepareThrustClassificationDatasets(
        yaml_cfg,
        data_cfg,
        output_np=True,
        vel_noise_std=args.velNoise,
        pos_noise_std=1e3 * args.velNoise,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    return (
        np.asarray(train_data),
        np.asarray(train_label).reshape(-1),
        np.asarray(val_data),
        np.asarray(val_label).reshape(-1),
        np.asarray(test_data),
        np.asarray(test_label).reshape(-1),
    )


def build_regression_windows(trajs, labels, lookback):
    # trajs: (N, T, D), labels: (N,)
    n, t, d = trajs.shape
    if t <= lookback:
        raise ValueError(f"Need T > lookback, got T={t}, lookback={lookback}")

    windows = []
    cls_targets = []
    reg_targets = []

    for i in range(n):
        c = int(labels[i])
        for start in range(0, t - lookback):
            end = start + lookback
            windows.append(trajs[i, start:end, :])
            reg_targets.append(trajs[i, end, :])
            cls_targets.append(c)

    x = np.asarray(windows, dtype=np.float32)
    y_cls = np.asarray(cls_targets, dtype=np.int64)
    y_reg = np.asarray(reg_targets, dtype=np.float32)
    return x, y_cls, y_reg


def split_trajectory_halves(trajs):
    # split into first half (conditioning/training) and second half (rollout target)
    t = trajs.shape[1]
    split_idx = t // 2
    if split_idx < 2:
        raise ValueError(f"Trajectory length too short to split in half (T={t}).")
    first_half = trajs[:, :split_idx, :]
    second_half = trajs[:, split_idx:, :]
    return first_half, second_half, split_idx


def fit_normalizer(train_x):
    d = train_x.shape[-1]
    mu = train_x.reshape(-1, d).mean(axis=0)
    sig = train_x.reshape(-1, d).std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    return NormState(
        mu=torch.tensor(mu, dtype=torch.float32),
        sig=torch.tensor(sig, dtype=torch.float32),
    )


def normalize_windows(x, y_reg, norm_state):
    mu = norm_state.mu.numpy()
    sig = norm_state.sig.numpy()
    x_n = (x - mu) / sig
    y_n = (y_reg - mu) / sig
    return x_n, y_n


def normalize_trajectories(trajs, norm_state):
    mu = norm_state.mu.numpy()
    sig = norm_state.sig.numpy()
    return (trajs - mu) / sig


def evaluate_trajectory_classification(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    cls_correct = 0
    loss_cls_sum = 0.0

    with torch.no_grad():
        for xb, yb_cls in loader:
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)

            logits, _, _ = model(xb)
            loss_cls = ce(logits, yb_cls)

            pred_cls = torch.argmax(logits, dim=1)
            cls_correct += (pred_cls == yb_cls).sum().item()
            total += yb_cls.numel()
            loss_cls_sum += loss_cls.item() * yb_cls.numel()

    cls_acc = cls_correct / max(1, total)
    cls_loss = loss_cls_sum / max(1, total)
    return cls_loss, cls_acc


def evaluate_window_regression(model, loader, device):
    model.eval()
    reg_se_sum = 0.0
    reg_count = 0

    with torch.no_grad():
        for xb, _, yb_reg in loader:
            xb = xb.to(device)
            yb_reg = yb_reg.to(device)
            _, _, mixed_pred = model(xb)
            reg_se_sum += torch.sum((mixed_pred - yb_reg) ** 2).item()
            reg_count += yb_reg.numel()

    return np.sqrt(reg_se_sum / max(1, reg_count))


def predict_class_from_sliding_windows(model, trajectory, norm_state, lookback, device):
    t = trajectory.shape[0]
    if t < lookback:
        raise ValueError("Not enough sequence length for sliding-window classification.")

    mu = norm_state.mu.to(device)
    sig = norm_state.sig.to(device)
    traj_t = torch.tensor(trajectory, dtype=torch.float32, device=device)
    traj_t = (traj_t - mu) / sig

    logits_list = []
    with torch.no_grad():
        for start in range(0, t - lookback + 1):
            window = traj_t[start : start + lookback, :].unsqueeze(0)
            logits, _, _ = model(window)
            logits_list.append(logits.squeeze(0))

    mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return int(torch.argmax(mean_logits).item())


def rollout_point_cloud(model, trajectories, labels, norm_state, lookback, split_idx, rollout_steps, device):
    # trajectories: (N, T, D), labels: (N,)
    model.eval()
    mu = norm_state.mu.to(device)
    sig = norm_state.sig.to(device)

    n, t, d = trajectories.shape
    if split_idx < lookback:
        raise ValueError("Not enough sequence length in first half for rollout seed window.")
    if split_idx + rollout_steps > t:
        raise ValueError("Requested rollout extends beyond trajectory length.")

    pred_states = []
    pred_classes = []
    true_classes = labels.astype(np.int64)

    with torch.no_grad():
        for i in range(n):
            pred_c = predict_class_from_sliding_windows(
                model=model,
                trajectory=trajectories[i, :split_idx, :],
                norm_state=norm_state,
                lookback=lookback,
                device=device,
            )
            pred_classes.append(pred_c)

            # seed rollout from the end of the first half so predictions target the second half
            seed = torch.tensor(
                trajectories[i, split_idx - lookback : split_idx, :],
                dtype=torch.float32,
                device=device,
            )
            seed = (seed - mu) / sig
            window = seed.unsqueeze(0)  # (1, lookback, D)

            for _ in range(rollout_steps):
                _, _, mixed_pred = model(window)  # normalized prediction
                pred_denorm = mixed_pred * sig + mu
                pred_states.append(pred_denorm.squeeze(0).cpu().numpy())

                next_norm = mixed_pred.unsqueeze(1)
                window = torch.cat([window[:, 1:, :], next_norm], dim=1)

    pred_states = np.asarray(pred_states, dtype=np.float32)  # (N*rollout_steps, D)
    pred_classes = np.asarray(pred_classes, dtype=np.int64)

    class_clouds = {}
    for c in range(4):
        idx = np.where(pred_classes == c)[0]
        if idx.size == 0:
            class_clouds[c] = np.zeros((0, rollout_steps, d), dtype=np.float32)
        else:
            cloud = pred_states.reshape(n, rollout_steps, d)[idx]
            class_clouds[c] = cloud

    return pred_states.reshape(n, rollout_steps, d), pred_classes, true_classes, class_clouds


def plot_position_clouds(pred_rollouts, pred_classes, save_path):
    # pred_rollouts: (N, K, 6)
    colors = ["black", "tab:blue", "tab:orange", "tab:red"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for c in range(4):
        idx = np.where(pred_classes == c)[0]
        if idx.size == 0:
            continue
        # only plot the final point of each rollout for clarity
        pts = pred_rollouts[idx, -1, :3].reshape(-1, 3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, alpha=0.2, c=colors[c], label=CLASS_LABELS[c])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Reachability Point Cloud (Position)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def alpha_shape_faces_and_volume(points, edge_quantile=0.95):
    n = points.shape[0]
    if n < 5:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    try:
        tet = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    simplices = tet.simplices  # (m, 4)
    p = points[simplices]      # (m, 4, 3)

    e01 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e02 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    e03 = np.linalg.norm(p[:, 3] - p[:, 0], axis=1)
    e12 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e13 = np.linalg.norm(p[:, 3] - p[:, 1], axis=1)
    e23 = np.linalg.norm(p[:, 3] - p[:, 2], axis=1)
    max_edge = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tet_vol = np.abs(np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0])) / 6.0
    valid = tet_vol > 1e-14
    if not np.any(valid):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    thresh = np.quantile(max_edge[valid], edge_quantile)
    keep = valid & (max_edge <= thresh)
    if not np.any(keep):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    kept = simplices[keep]
    faces = np.concatenate(
        [
            kept[:, [0, 1, 2]],
            kept[:, [0, 1, 3]],
            kept[:, [0, 2, 3]],
            kept[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces_sorted = np.sort(faces, axis=1)
    uniq_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = uniq_faces[counts == 1]
    return points[boundary_faces], tet_vol[keep].sum()


def main(args):
    set_seed(args.seed)

    device = getDevice()
    test_orbit = args.test if args.test is not None else args.orbit
    plot_tag = f"{args.propMin}min_train-{args.orbit}_test-{test_orbit}" if not args.OE else f"{args.propMin}min_train-{args.orbit}_test-{test_orbit}_OE"
    model_tag = f"{args.propMin}min_train-{args.orbit}" if not args.OE else f"{args.propMin}min_train-{args.orbit}_OE"

    train_traj, train_lbl, val_traj, val_lbl, test_traj, test_lbl = load_classification_trajectories(args)

    train_traj_first, train_traj_second, train_split = split_trajectory_halves(train_traj)
    val_traj_first, val_traj_second, val_split = split_trajectory_halves(val_traj)
    test_traj_first, test_traj_second, test_split = split_trajectory_halves(test_traj)
    if not (train_split == val_split == test_split):
        raise ValueError("Train/val/test trajectories do not share the same half-split index.")
    half_idx = train_split

    if train_traj_first.shape[1] <= args.lookback:
        raise ValueError(
            f"lookback ({args.lookback}) must be smaller than first-half length ({train_traj_first.shape[1]})."
        )
    print(
        f"Using trajectory half-split at index {half_idx}: "
        f"first half length={train_traj_first.shape[1]}, second half length={test_traj_second.shape[1]} "
        f"(rollout length forced to second half)."
    )

    xtr, ytr_cls, ytr_reg = build_regression_windows(train_traj_first, train_lbl, args.lookback)
    xva, yva_cls, yva_reg = build_regression_windows(val_traj_first, val_lbl, args.lookback)
    xte, yte_cls, yte_reg = build_regression_windows(test_traj_first, test_lbl, args.lookback)

    norm_state = fit_normalizer(xtr)
    xtr, ytr_reg = normalize_windows(xtr, ytr_reg, norm_state)
    xva, yva_reg = normalize_windows(xva, yva_reg, norm_state)
    xte, yte_reg = normalize_windows(xte, yte_reg, norm_state)
    train_traj_n = normalize_trajectories(train_traj_first, norm_state)
    val_traj_n = normalize_trajectories(val_traj_first, norm_state)
    test_traj_n = normalize_trajectories(test_traj_first, norm_state)

    train_ds = JointWindowDataset(
        torch.tensor(xtr, dtype=torch.float32),
        torch.tensor(ytr_cls, dtype=torch.long),
        torch.tensor(ytr_reg, dtype=torch.float32),
    )
    val_ds = JointWindowDataset(
        torch.tensor(xva, dtype=torch.float32),
        torch.tensor(yva_cls, dtype=torch.long),
        torch.tensor(yva_reg, dtype=torch.float32),
    )
    test_ds = JointWindowDataset(
        torch.tensor(xte, dtype=torch.float32),
        torch.tensor(yte_cls, dtype=torch.long),
        torch.tensor(yte_reg, dtype=torch.float32),
    )
    train_cls_ds = TrajectoryDataset(
        torch.tensor(train_traj_n, dtype=torch.float32),
        torch.tensor(train_lbl, dtype=torch.long),
    )
    val_cls_ds = TrajectoryDataset(
        torch.tensor(val_traj_n, dtype=torch.float32),
        torch.tensor(val_lbl, dtype=torch.long),
    )
    test_cls_ds = TrajectoryDataset(
        torch.tensor(test_traj_n, dtype=torch.float32),
        torch.tensor(test_lbl, dtype=torch.long),
    )

    train_reg_loader = data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_reg_loader = data.DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=True)
    test_reg_loader = data.DataLoader(test_ds, batch_size=args.batch, shuffle=False, pin_memory=True)
    train_cls_loader = data.DataLoader(train_cls_ds, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_cls_loader = data.DataLoader(val_cls_ds, batch_size=args.batch, shuffle=False, pin_memory=True)
    test_cls_loader = data.DataLoader(test_cls_ds, batch_size=args.batch, shuffle=False, pin_memory=True)

    input_size = train_traj.shape[-1]
    num_classes = 4
    model = LSTMThrustReachabilityEnsemble(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        output_size=input_size,
        dropout=args.dropout,
    ).to(device)

    printModelParmSize(model)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_path = os.path.join(args.model_dir, f"{model_tag}_lstm_ensemble_best.pt")

    load_model = args.load_checkpoint is not None and os.path.exists(args.load_checkpoint)

    if load_model:
        print(f"Loading existing checkpoint from {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        norm_state = NormState(mu=ckpt["mu"], sig=ckpt["sig"])
        print("Checkpoint loaded successfully. Skipping training.")
    else:
        trainTimer = timer()
        for epoch in range(args.epochs):
            model.train()
            running_cls = 0.0
            running_reg = 0.0
            seen_cls = 0
            seen_reg = 0

            for xb, yb_cls in train_cls_loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)

                opt.zero_grad(set_to_none=True)
                logits, _, _ = model(xb)
                loss_cls = ce(logits, yb_cls)
                loss_cls.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()

                running_cls += loss_cls.item() * xb.size(0)
                seen_cls += xb.size(0)

            for xb, yb_cls, yb_reg in train_reg_loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)
                yb_reg = yb_reg.to(device)

                opt.zero_grad(set_to_none=True)
                _, expert_preds, _ = model(xb)

                expert_true = expert_preds[torch.arange(xb.size(0), device=device), yb_cls]  # (B, D)
                loss_reg = mse(expert_true, yb_reg)
                loss = args.lambda_reg * loss_reg
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()

                running_reg += loss_reg.item() * xb.size(0)
                seen_reg += xb.size(0)

            tr_cls_loss = running_cls / max(1, seen_cls)
            tr_reg_loss = running_reg / max(1, seen_reg)
            va_cls_loss, va_cls_acc = evaluate_trajectory_classification(model, val_cls_loader, device)
            va_reg_rmse = evaluate_window_regression(model, val_reg_loader, device)
            joint_val = va_cls_loss + args.lambda_reg * (va_reg_rmse ** 2)
            sched.step(joint_val)

            if joint_val < best_val:
                best_val = joint_val
                torch.save(
                    {
                        "model": model.state_dict(),
                        "mu": norm_state.mu,
                        "sig": norm_state.sig,
                        "args": vars(args),
                    },
                    best_path,
                )

            lr_now = opt.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | train_cls_loss={tr_cls_loss:.5f} | train_reg_loss={tr_reg_loss:.5f} | "
                f"val_cls_acc={100.0 * va_cls_acc:.2f}% | val_reg_rmse={va_reg_rmse:.5f} | lr={lr_now:.2e}"
            )
        trainTimer.toc()
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    testTimer = timer()
    te_cls_loss, te_cls_acc = evaluate_trajectory_classification(model, test_cls_loader, device)
    te_reg_rmse = evaluate_window_regression(model, test_reg_loader, device)
    testTimer.toc()
    print("\nFinal Test Metrics")
    print(f"Classification Loss: {te_cls_loss:.5f}")
    print(f"Classification Accuracy: {100.0 * te_cls_acc:.2f}%")
    print(f"Regression RMSE (normalized): {te_reg_rmse:.5f}")

    # generate confusion matrix for test set
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, yb_cls in test_cls_loader:
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            logits, _, _ = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb_cls.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    pred_labels = torch.argmax(all_logits, dim=1)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels.numpy(), pred_labels.numpy(), labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Test Set Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"{plot_tag}_test_confusion_matrix.png"))
    plt.close()

    pred_rollouts, pred_classes, true_classes, class_clouds = rollout_point_cloud(
        model,
        trajectories=test_traj,
        labels=test_lbl,
        norm_state=norm_state,
        lookback=args.lookback,
        split_idx=half_idx,
        rollout_steps=test_traj_second.shape[1],
        device=device,
    )

    true_rollout_targets = test_traj_second

    # print per class rollout counts
    for c in range(4):
        count = np.sum(pred_classes == c)
        print(f"Class {CLASS_LABELS[c]}: {count} rollouts")
    # plot per predicted class rollout clouds with classified subset in black
    for c in range(4):
        idx_pred = np.where(pred_classes == c)[0]
        if idx_pred.size == 0:
            continue

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # plot true final states for trajectories classified into class c
        true_pts = true_rollout_targets[idx_pred, -1, :3]
        ax.scatter(true_pts[:, 0], true_pts[:, 1], true_pts[:, 2], s=3, alpha=0.2, c="black", label="Classified Subset (True)")

        # plot predicted final states in color
        pred_pts = pred_rollouts[idx_pred, -1, :3]
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], s=3, alpha=0.2, c="tab:blue", label="Classified Subset (Pred)")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Class {CLASS_LABELS[c]} Reachability Point Cloud (Position)")
        ax.legend(loc="best")
        plt.tight_layout()
        class_name_file = CLASS_LABELS[c].lower().replace(" ", "_")
        plt.savefig(os.path.join(args.save_dir, f"{plot_tag}_class-{class_name_file}_point_cloud.png"))
        plt.close()

    cloud_path = os.path.join(args.save_dir, "predicted_reachability_point_clouds.npz")
    np.savez(
        cloud_path,
        pred_rollouts=pred_rollouts,
        pred_classes=pred_classes,
        true_classes=true_classes,
        class0=class_clouds[0],
        class1=class_clouds[1],
        class2=class_clouds[2],
        class3=class_clouds[3],
        class_labels=np.asarray(CLASS_LABELS, dtype=object),
    )
    print(f"Saved point cloud arrays to: {cloud_path}")

    fig_path = os.path.join(args.save_dir, f"{plot_tag}_predicted_position_point_cloud.png")
    plot_position_clouds(pred_rollouts, pred_classes, fig_path)
    # plot true classes for reference
    plot_position_clouds(
        true_rollout_targets,
        test_lbl,
        os.path.join(args.save_dir, f"{plot_tag}_true_position_point_cloud.png"),
    )
    print(f"Saved point cloud figure to: {fig_path}")

    # compute and save alpha-shape plots per predicted class
    for c in range(4):
        class_name = CLASS_LABELS[c]
        class_name_file = class_name.lower().replace(" ", "_")
        idx_pred = np.where(pred_classes == c)[0]

        if idx_pred.size < 5:
            print(
                f"Skipping alpha-shape for class {class_name}: "
                f"need >=5 trajectories in predicted set (pred={idx_pred.size})."
            )
            continue

        pos_true = true_rollout_targets[idx_pred, -1, :3]
        pos_pred = pred_rollouts[idx_pred, -1, :3]
        vel_true = true_rollout_targets[idx_pred, -1, 3:]
        vel_pred = pred_rollouts[idx_pred, -1, 3:]

        try:
            pos_faces_true, pos_vol_true = alpha_shape_faces_and_volume(pos_true, edge_quantile=0.95)
            pos_faces_pred, pos_vol_pred = alpha_shape_faces_and_volume(pos_pred, edge_quantile=0.95)
            vel_faces_true, vel_vol_true = alpha_shape_faces_and_volume(vel_true, edge_quantile=0.95)
            vel_faces_pred, vel_vol_pred = alpha_shape_faces_and_volume(vel_pred, edge_quantile=0.95)
        except QhullError:
            print(f"Skipping alpha-shape for class {class_name}: hull construction failed.")
            continue

        pos_ratio = float(pos_vol_pred) / float(pos_vol_true) if pos_vol_true > 0 else float("inf")
        vel_ratio = float(vel_vol_pred) / float(vel_vol_true) if vel_vol_true > 0 else float("inf")
        print(
            f"[{class_name}] Pos Vol True={pos_vol_true:.4f}, Pred={pos_vol_pred:.4f}, Ratio={pos_ratio:.4f}"
        )
        print(
            f"[{class_name}] Vel Vol True={vel_vol_true:.4f}, Pred={vel_vol_pred:.4f}, Ratio={vel_ratio:.4f}"
        )

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        for ax, t_pts, p_pts, t_faces, p_faces, title, labels in [
            (
                ax1,
                pos_true,
                pos_pred,
                pos_faces_true,
                pos_faces_pred,
                f"Position Alpha Shapes ({class_name}, Ratio {pos_ratio:.4f})",
                ("x", "y", "z"),
            ),
            (
                ax2,
                vel_true,
                vel_pred,
                vel_faces_true,
                vel_faces_pred,
                f"Velocity Alpha Shapes ({class_name}, Ratio {vel_ratio:.4f})",
                ("vx", "vy", "vz"),
            ),
        ]:
            ax.scatter(t_pts[:, 0], t_pts[:, 1], t_pts[:, 2], s=3, alpha=0.2, c="k")
            ax.scatter(p_pts[:, 0], p_pts[:, 1], p_pts[:, 2], s=3, alpha=0.2, c="r")
            ax.add_collection3d(Poly3DCollection(t_faces, facecolor="k", alpha=0.08, edgecolor="none"))
            ax.add_collection3d(Poly3DCollection(p_faces, facecolor="r", alpha=0.08, edgecolor="none"))
            all_pts = np.vstack((t_pts, p_pts))
            # mins = all_pts.min(axis=0)
            # maxs = all_pts.max(axis=0)
            # spans = np.maximum(maxs - mins, 1e-6)
            # ax.set_box_aspect(spans)
            # ax.set_xlim(mins[0], maxs[0])
            # ax.set_ylim(mins[1], maxs[1])
            # ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            ax.set_title(title)
            # create custom legend
            ax.scatter([], [], [], c="k", alpha=0.2, label="True Trajectories")
            ax.scatter([], [], [], c="r", alpha=0.2, label="Predicted Trajectories")
            ax.legend(loc="best")
        fig.suptitle(f"Final-State 3D Alpha Shapes: {class_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.save_dir,
                f"{plot_tag}_final_state_alpha_shapes_3d_{class_name_file}.png",
            )
        )
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.save_dir, exist_ok=True)

    log = os.path.join(args.save_dir, "log.txt")
    with open(log, 'w', buffering=1, encoding='utf-8') as f, \
        redirect_stdout(f), redirect_stderr(f):
        main(args)
