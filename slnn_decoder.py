"""
slnn_decoder.py — Deep SLNN Decoder Module cho hệ thống 7/9-Rate Sparse Code STT-MRAM
==================================================================================
Triển khai bằng PyTorch nhưng giữ nguyên API public để các file mô phỏng hiện có
vẫn dùng được:
  - train_slnn()
  - decode_slnn_batch()
  - save_slnn()
  - load_slnn()

Hỗ trợ hai chế độ đầu ra:
  - softmax: 128 class, phù hợp với decode bằng bảng tra `_decode_table`
  - sigmoid: 9 bit độc lập, sau đó lookup codebook bằng Euclidean distance

Checkpoint mới được lưu bằng `torch.save`, nhưng `load_slnn()` vẫn đọc được file
`.npy` cũ được tạo bởi phiên bản NumPy trước đây.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def _as_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _as_long_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long)


# ───────────────────────────────────────────────────────────────────────────
# KIẾN TRÚC MẠNG DEEP SLNN
# ───────────────────────────────────────────────────────────────────────────
class DeepSLNNNumpy(nn.Module):
    """PyTorch implementation of the SLNN decoder, kept under the old class name for compatibility."""

    def __init__(
        self,
        input_size: int = 9,
        h1: int = 128,
        h2: int = 64,
        output_size: int = 128,
        lr: float = 0.01,
        output_mode: str = "softmax",
    ):
        super().__init__()

        if output_mode not in {"softmax", "sigmoid"}:
            raise ValueError("output_mode phải là 'softmax' hoặc 'sigmoid'")

        self.input_size = input_size
        self.h1 = h1
        self.h2 = h2
        self.output_size = output_size
        self.lr = lr
        self.output_mode = output_mode

        self.fc1 = nn.Linear(input_size, h1)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(h2, output_size)

        self._reset_parameters()

        self.norm_mu: float = 0.0
        self.norm_std: float = 1.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.fc3.bias)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return self.fc3(x)

    def forward(self, x):
        x_tensor = _as_float_tensor(x) if not torch.is_tensor(x) else x.float()
        logits = self._logits(x_tensor)
        if self.output_mode == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def compute_loss(self, probs, labels) -> float:
        probs_tensor = _as_float_tensor(probs) if not torch.is_tensor(probs) else probs.float()
        if self.output_mode == "sigmoid":
            labels_tensor = _as_float_tensor(labels)
            eps = 1e-12
            loss = -torch.mean(
                labels_tensor * torch.log(probs_tensor + eps)
                + (1.0 - labels_tensor) * torch.log(1.0 - probs_tensor + eps)
            )
        else:
            labels_tensor = _as_long_tensor(labels)
            loss = -torch.mean(
                torch.log(probs_tensor[torch.arange(labels_tensor.shape[0]), labels_tensor] + 1e-12)
            )
        return float(loss.item())

    def backward(self, labels) -> None:
        raise NotImplementedError("PyTorch version uses autograd; gọi backward() trực tiếp không còn cần thiết.")

    def predict(self, x):
        """Softmax mode: trả về class index (batch,). Sigmoid mode: trả về codeword bit (batch, 9)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            if self.output_mode == "sigmoid":
                return (out >= 0.5).to(torch.int64).cpu().numpy()
            return torch.argmax(out, dim=1).cpu().numpy()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Áp dụng Z-score dùng mu/std đã lưu."""
        return (x - self.norm_mu) / (self.norm_std + 1e-8)


# ───────────────────────────────────────────────────────────────────────────
# HÀM TRAIN
# ───────────────────────────────────────────────────────────────────────────
def train_slnn(
    codebook: np.ndarray,
    sigma_mu,
    P1: float,
    nr_train: int = 1_500_000,
    h1: int = 128,
    h2: int = 64,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.01,
    patience: int = 5,
    lr_decay_every: int = 10,
    lr_decay_rate: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
    output_mode: str = "softmax",
    channel_fn=None,
) -> DeepSLNNNumpy:
    """Train mô hình SLNN bằng PyTorch autograd."""
    if channel_fn is None:
        from config import MRAM_channel_batch as channel_fn

    sigma_list = sigma_mu if isinstance(sigma_mu, (list, tuple)) else [sigma_mu]
    out_size = 9 if output_mode == "sigmoid" else 128

    if verbose:
        print(f"\n{'='*66}")
        print(f"[DEEP SLNN TRAIN] output_mode={output_mode}")
        print(f"  sigma={[f'{s*100:.0f}%' for s in sigma_list]}, P1={P1:.0e}")
        print(f"  nr_train={nr_train:,} | H1={h1}, H2={h2} | epochs={epochs}")
        print(f"  batch={batch_size} | lr={lr} | decay x{lr_decay_rate} every {lr_decay_every} epoch")
        print(f"{'='*66}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    per = nr_train // len(sigma_list)
    X_parts, y_parts = [], []

    for sm in sigma_list:
        lbl = np.random.randint(0, 128, per)
        rx = channel_fn(codebook[lbl], sm, P1)
        X_parts.append(rx)
        if output_mode == "sigmoid":
            y_parts.append(codebook[lbl].astype(np.float32))
        else:
            y_parts.append(lbl.astype(np.int64))

    X_raw = np.concatenate(X_parts, axis=0)
    labels = np.concatenate(y_parts, axis=0)

    norm_mu = float(X_raw.mean())
    norm_std = float(X_raw.std())
    X = (X_raw - norm_mu) / (norm_std + 1e-8)

    if verbose:
        print(
            f"\n  [Norm] mu={norm_mu:.4f}, std={norm_std:.4f}  "
            f"(old hardcode: mu=1.5, std=0.5)"
        )

    perm = np.random.permutation(len(X))
    X = X[perm]
    labels = labels[perm]

    split = int(0.9 * len(X))
    X_tr, y_tr = X[:split], labels[:split]
    X_val, y_val = X[split:], labels[split:]

    model = DeepSLNNNumpy(
        input_size=9,
        h1=h1,
        h2=h2,
        output_size=out_size,
        lr=lr,
        output_mode=output_mode,
    )
    model.norm_mu = norm_mu
    model.norm_std = norm_std

    device = torch.device("cpu")
    model.to(device)

    if output_mode == "sigmoid":
        criterion = nn.BCEWithLogitsLoss()
        y_tr_tensor = _as_float_tensor(y_tr)
        y_val_tensor = _as_float_tensor(y_val)
    else:
        criterion = nn.CrossEntropyLoss()
        y_tr_tensor = _as_long_tensor(y_tr)
        y_val_tensor = _as_long_tensor(y_val)

    X_tr_tensor = _as_float_tensor(X_tr)
    X_val_tensor = _as_float_tensor(X_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay_rate)
        if lr_decay_every > 0
        else None
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    if verbose:
        print(f"\n{'Epoch':>8} | {'Train Loss':>12} | {'Val Loss':>10} | {'Val Acc':>8} | {'LR':>8} | Note")
        print(f"  {'-'*70}")

    for ep in range(epochs):
        model.train()
        perm_ep = torch.randperm(X_tr_tensor.shape[0])
        X_s = X_tr_tensor[perm_ep]
        y_s = y_tr_tensor[perm_ep]

        train_loss = 0.0
        n_batches = 0

        for i in range(0, X_s.shape[0], batch_size):
            xb = X_s[i : i + batch_size].to(device)
            yb = y_s[i : i + batch_size].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model._logits(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model._logits(X_val_tensor.to(device))
            val_loss = float(criterion(val_logits, y_val_tensor.to(device)).item())
            val_probs = torch.sigmoid(val_logits) if output_mode == "sigmoid" else torch.softmax(val_logits, dim=1)

            if output_mode == "sigmoid":
                val_acc = float(
                    torch.all((val_probs >= 0.5).to(torch.int64) == y_val_tensor.to(device).to(torch.int64), dim=1)
                    .float()
                    .mean()
                    .item()
                )
            else:
                val_acc = float((torch.argmax(val_probs, dim=1) == y_val_tensor.to(device)).float().mean().item())

        model.history["train_loss"].append(avg_train)
        model.history["val_loss"].append(val_loss)
        model.history["val_acc"].append(val_acc)

        note = ""
        if val_loss - avg_train > 0.5:
            note = "large gap"
        elif val_acc > 0.999:
            note = "converged"

        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(
                f"  {ep+1:6d}/{epochs} | {avg_train:12.4f} | {val_loss:10.4f} | "
                f"{val_acc*100:7.1f}% | {current_lr:8.5f} | {note}"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"\n  Early stop at epoch {ep+1}")
                break

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(
            f"\n  [DONE] Best val_loss={best_val_loss:.4f} | "
            f"norm_mu={model.norm_mu:.4f} | norm_std={model.norm_std:.4f}"
        )
    return model


# ───────────────────────────────────────────────────────────────────────────
# HÀM GIẢI MÃ
# ───────────────────────────────────────────────────────────────────────────
def decode_slnn_batch(model: DeepSLNNNumpy, received_raw: np.ndarray, in_bits: np.ndarray) -> tuple:
    """Giải mã batch nhận được."""
    from config import _decode_table, codebook_7b9b

    x_norm = model.normalize(received_raw)

    if model.output_mode == "sigmoid":
        pred_cw = model.predict(x_norm)
        dists = np.sum((pred_cw[:, np.newaxis, :] - codebook_7b9b[np.newaxis, :, :]) ** 2, axis=2)
        best = np.argmin(dists, axis=1)
    else:
        best = model.predict(x_norm)

    out_dec = _decode_table[best]
    errs = np.sum(out_dec != in_bits, axis=1)
    return int(np.sum(errs)), int(np.sum(errs > 0))


# ───────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH LƯU / TẢI
# ───────────────────────────────────────────────────────────────────────────
def save_slnn(model: DeepSLNNNumpy, path: str) -> None:
    payload = {
        "format": "torch_slnn_v1",
        "input_size": model.input_size,
        "h1": model.h1,
        "h2": model.h2,
        "output_size": model.output_size,
        "lr": model.lr,
        "norm_mu": model.norm_mu,
        "norm_std": model.norm_std,
        "output_mode": model.output_mode,
        "history": model.history,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    print(f"[SLNN] Saved model -> {path}")


def _load_numpy_checkpoint(path: str) -> DeepSLNNNumpy:
    data = dict(np.load(path, allow_pickle=True).item())
    model = DeepSLNNNumpy(
        input_size=data["W1"].shape[0],
        h1=data["W1"].shape[1],
        h2=data["W2"].shape[1],
        output_size=data["W3"].shape[1],
        lr=float(data["lr"]),
        output_mode=str(data.get("output_mode", "softmax")),
    )

    with torch.no_grad():
        model.fc1.weight.copy_(torch.as_tensor(data["W1"].T, dtype=torch.float32))
        model.fc1.bias.copy_(torch.as_tensor(data["b1"], dtype=torch.float32))
        model.fc2.weight.copy_(torch.as_tensor(data["W2"].T, dtype=torch.float32))
        model.fc2.bias.copy_(torch.as_tensor(data["b2"], dtype=torch.float32))
        model.fc3.weight.copy_(torch.as_tensor(data["W3"].T, dtype=torch.float32))
        model.fc3.bias.copy_(torch.as_tensor(data["b3"], dtype=torch.float32))

    model.norm_mu = float(data.get("norm_mu", 1.5))
    model.norm_std = float(data.get("norm_std", 0.5))
    model.history = data.get("history", {"train_loss": [], "val_loss": [], "val_acc": []})
    print(
        f"[SLNN] Loaded model <- {path}  "
        f"(norm_mu={model.norm_mu:.4f}, norm_std={model.norm_std:.4f})"
    )
    return model


def load_slnn(path: str) -> DeepSLNNNumpy:
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return _load_numpy_checkpoint(path)

    if isinstance(data, dict) and data.get("format") == "torch_slnn_v1":
        model = DeepSLNNNumpy(
            input_size=int(data.get("input_size", 9)),
            h1=int(data.get("h1", 128)),
            h2=int(data.get("h2", 64)),
            output_size=int(data.get("output_size", 128)),
            lr=float(data.get("lr", 0.01)),
            output_mode=str(data.get("output_mode", "softmax")),
        )
        model.load_state_dict(data["state_dict"])
        model.norm_mu = float(data.get("norm_mu", 1.5))
        model.norm_std = float(data.get("norm_std", 0.5))
        model.history = data.get("history", {"train_loss": [], "val_loss": [], "val_acc": []})
        print(
            f"[SLNN] Loaded model <- {path}  "
            f"(norm_mu={model.norm_mu:.4f}, norm_std={model.norm_std:.4f})"
        )
        return model

    return _load_numpy_checkpoint(path)