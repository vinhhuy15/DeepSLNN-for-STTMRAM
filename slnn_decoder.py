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
# KIẾN TRÚC MẠNG BASELINE SLNN (9 -> 128 -> 128 -> 128)
# ───────────────────────────────────────────────────────────────────────────
class DeepSLNNNumpy(nn.Module):
    def __init__(
        self,
        input_size: int = 9,
        h1: int = 128,
        h2: int = 128,
        output_size: int = 128,
        lr: float = 0.01,
        output_mode: str = "softmax",
    ):
        super().__init__()

        self.input_size = input_size
        self.h1 = h1
        self.h2 = h2
        self.output_size = output_size
        self.lr = lr
        self.output_mode = "softmax"

        self.fc1 = nn.Linear(input_size, h1)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc3 = nn.Linear(h2, output_size)

        self._reset_parameters()

        self.norm_mu: float = 1.5
        self.norm_std: float = 0.5
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
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return torch.argmax(out, dim=1).cpu().numpy()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.norm_mu) / (self.norm_std + 1e-8)


# ───────────────────────────────────────────────────────────────────────────
# 3. HÀM TRAIN (CÓ DATA AUGMENTATION & ATTENUATOR)
# ───────────────────────────────────────────────────────────────────────────
def train_slnn(
    codebook: np.ndarray,
    sigma_mu,
    P1: float,
    nr_train: int = 1_500_000,
    h1: int = 128,
    h2: int = 128,
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
    if channel_fn is None:
        from config import MRAM_channel_batch as channel_fn

    sigma_list = sigma_mu if isinstance(sigma_mu, (list, tuple)) else [sigma_mu]

    if verbose:
        print(f"\n{'='*70}")
        print(f"[DEEP SLNN TRAIN] BASELINE NETWORK (9->128->128->128)")
        print(f"  Loss: CrossEntropy")
        print(f"  Data Augmentation: offset_mu (-0.25 to -0.15) & offset_sigma (0~0.05)")
        print(f"  No ALPHA for SLNN (Learn raw data)")
        print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    per = nr_train // len(sigma_list)
    X_parts, y_parts = [], []
    chunk_size = 10000 

    for sm in sigma_list:
        for i in range(0, per, chunk_size):
            current_chunk = min(chunk_size, per - i)
            lbl = np.random.randint(0, 128, current_chunk)

            random_offset_mu = np.random.uniform(-0.25, -0.15)
            random_offset_sigma = np.random.uniform(0.0, 0.05)

            rx = channel_fn(
                codebook[lbl], 
                sm, 
                P1,
                offset_mu=random_offset_mu,
                offset_sigma_ratio=random_offset_sigma
            )

            X_parts.append(rx)
            y_parts.append(lbl.astype(np.int64))

    X_raw = np.concatenate(X_parts, axis=0)
    labels = np.concatenate(y_parts, axis=0)

    X = (X_raw - 1.5) / 0.5

    perm = np.random.permutation(len(X))
    X = X[perm]
    labels = labels[perm]

    split = int(0.9 * len(X))
    X_tr, y_tr = X[:split], labels[:split]
    X_val, y_val = X[split:], labels[split:]

    model = DeepSLNNNumpy(input_size=9, h1=h1, h2=h2, output_size=128, lr=lr, output_mode="softmax")
    model.norm_mu = 1.5
    model.norm_std = 0.5
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    y_tr_tensor = _as_long_tensor(y_tr)
    y_val_tensor = _as_long_tensor(y_val)
    X_tr_tensor = _as_float_tensor(X_tr)
    X_val_tensor = _as_float_tensor(X_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay_rate) if lr_decay_every > 0 else None

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

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
            val_probs = torch.softmax(val_logits, dim=1)
            val_acc = float((torch.argmax(val_probs, dim=1) == y_val_tensor.to(device)).float().mean().item())

        model.history["train_loss"].append(avg_train)
        model.history["val_loss"].append(val_loss)
        model.history["val_acc"].append(val_acc)

        if verbose:
            print(f"  {ep+1:6d}/{epochs} | {avg_train:12.4f} | {val_loss:10.4f} | {val_acc*100:7.1f}%")

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
    return model


# ───────────────────────────────────────────────────────────────────────────
# 4. HÀM GIẢI MÃ
# ───────────────────────────────────────────────────────────────────────────
def decode_slnn_batch(model: DeepSLNNNumpy, received_raw: np.ndarray, in_bits: np.ndarray) -> tuple:
    from config import _decode_table

    x_norm = model.normalize(received_raw)
    best = model.predict(x_norm)
    out_dec = _decode_table[best]
    errs = np.sum(out_dec != in_bits, axis=1)
    return int(np.sum(errs)), int(np.sum(errs > 0))

# ───────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH LƯU / TẢI
# ───────────────────────────────────────────────────────────────────────────
def save_slnn(model: DeepSLNNNumpy, path: str) -> None:
    payload = {
        "format": "torch_slnn_v3",
        "input_size": model.input_size, "h1": model.h1, "h2": model.h2,
        "output_size": model.output_size, "lr": model.lr,
        "output_mode": model.output_mode, "history": model.history,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    print(f"[SLNN] Saved model -> {path}")

def load_slnn(path: str) -> DeepSLNNNumpy:
    data = torch.load(path, map_location="cpu")
    model = DeepSLNNNumpy(
        input_size=int(data.get("input_size", 9)), h1=int(data.get("h1", 128)),
        h2=int(data.get("h2", 128)), output_size=int(data.get("output_size", 128)),
        lr=float(data.get("lr", 0.01)), output_mode="softmax",
    )
    model.load_state_dict(data["state_dict"])
    model.norm_mu = 1.5
    model.norm_std = 0.5
    print(f"[SLNN] Loaded model <- {path}  (norm_mu={model.norm_mu:.4f}, norm_std={model.norm_std:.4f})")
    return model