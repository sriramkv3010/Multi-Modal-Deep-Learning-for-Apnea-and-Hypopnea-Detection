import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.cnn_model import SleepApneaCNN

EPOCHS = 60
BATCH_SIZE = 32
LR = 3e-4
PATIENCE = 10
N_SAMPLES = 960

# thresholds for minority class prediction
# lower than 0.5 so model predicts Apnea/Hypopnea even when not super confident
APNEA_THRESHOLD = 0.20
HYPOPNEA_THRESHOLD = 0.30

torch.manual_seed(42)
np.random.seed(42)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def augment_window(X):
    X = X.copy()

    # gaussian noise — stronger for rare events
    if np.random.rand() < 0.6:
        X += np.random.randn(*X.shape).astype(np.float32) * 0.08

    # amplitude scale
    if np.random.rand() < 0.6:
        X *= np.random.uniform(0.75, 1.25)

    # time shift
    if np.random.rand() < 0.5:
        X = np.roll(X, np.random.randint(-80, 80), axis=1)

    # random channel dropout (simulates sensor noise)
    if np.random.rand() < 0.2:
        ch = np.random.randint(0, 3)
        X[ch] = 0.0

    # sign flip on flow channel (sensor can be inverted)
    if np.random.rand() < 0.3:
        X[0] = -X[0]

    return X


def oversample(X, y, label_map):
    apnea_idx = np.where(y == label_map.get("Apnea", -1))[0]
    hypopnea_idx = np.where(y == label_map.get("Hypopnea", -1))[0]
    normal_idx = np.where(y == label_map.get("Normal", -1))[0]

    X_list, y_list = [X], [y]

    # bring Apnea up to 50% of Normal count
    if len(apnea_idx) > 0:
        target = len(normal_idx) // 2
        needed = max(0, target - len(apnea_idx))
        reps = int(np.ceil(needed / len(apnea_idx))) + 1
        for _ in range(reps):
            aug = np.array([augment_window(X[i]) for i in apnea_idx])
            X_list.append(aug)
            y_list.append(y[apnea_idx])

    # bring Hypopnea up to 70% of Normal count
    if len(hypopnea_idx) > 0:
        target = int(len(normal_idx) * 0.7)
        needed = max(0, target - len(hypopnea_idx))
        reps = int(np.ceil(needed / len(hypopnea_idx))) + 1
        for _ in range(reps):
            aug = np.array([augment_window(X[i]) for i in hypopnea_idx])
            X_list.append(aug)
            y_list.append(y[hypopnea_idx])

    X_out = np.concatenate(X_list, axis=0)
    y_out = np.concatenate(y_list, axis=0)
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def make_xy(df, label_map):
    flow = df[[f"flow_{i}" for i in range(N_SAMPLES)]].values.astype(np.float32)
    thorac = df[[f"thorac_{i}" for i in range(N_SAMPLES)]].values.astype(np.float32)
    spo2 = df[[f"spo2_{i}" for i in range(N_SAMPLES)]].values.astype(np.float32)
    X = np.stack([flow, thorac, spo2], axis=1)
    y = df["label"].map(label_map).values.astype(np.int64)
    return X, y


def predict_with_threshold(probs, label_map):
    # probs: (N, n_classes) numpy array of softmax probabilities
    # priority: Apnea > Hypopnea > Normal
    apnea_idx = label_map.get("Apnea", -1)
    hypopnea_idx = label_map.get("Hypopnea", -1)
    normal_idx = label_map.get("Normal", -1)

    preds = np.full(len(probs), normal_idx)

    if hypopnea_idx >= 0:
        preds[probs[:, hypopnea_idx] >= HYPOPNEA_THRESHOLD] = hypopnea_idx

    if apnea_idx >= 0:
        preds[probs[:, apnea_idx] >= APNEA_THRESHOLD] = apnea_idx

    return preds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_probs, all_true = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        out = model(X_batch)
        probs = F.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_true.extend(y_batch.numpy())

    return np.array(all_probs), np.array(all_true)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        out = model(X_batch)
        loss = criterion(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train_fold(
    fold_num, test_pid, train_df, test_df, label_map, n_classes, out_dir, device
):
    X_train_raw, y_train_raw = make_xy(train_df, label_map)
    X_test, y_test = make_xy(test_df, label_map)

    label_decode = {v: k for k, v in label_map.items()}

    print(f"\nFold {fold_num} | Test: {test_pid}")
    print(
        f"  Train before oversample: "
        + "  ".join(
            f"{label_decode[c]}={( y_train_raw==c).sum()}" for c in range(n_classes)
        )
    )

    X_train, y_train = oversample(X_train_raw, y_train_raw, label_map)

    print(
        f"  Train after  oversample: "
        + "  ".join(f"{label_decode[c]}={(y_train==c).sum()}" for c in range(n_classes))
    )

    val_size = max(1, int(0.15 * len(X_train)))
    train_size = len(X_train) - val_size
    idx = np.random.permutation(len(X_train))

    X_tr, y_tr = X_train[idx[:train_size]], y_train[idx[:train_size]]
    X_va, y_va = X_train[idx[train_size:]], y_train[idx[train_size:]]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SleepApneaCNN(n_classes=n_classes).to(device)

    counts = np.array([(y_tr == ci).sum() for ci in range(n_classes)], dtype=float)
    cw = np.zeros(n_classes)
    for ci in range(n_classes):
        if counts[ci] > 0:
            cw[ci] = len(y_tr) / (n_classes * counts[ci])
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)

    criterion = FocalLoss(gamma=2.0, weight=cw_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    patience_count = 0
    save_path = os.path.join(out_dir, f"fold{fold_num}_{test_pid}.pt")

    print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))

    # get probabilities for threshold-based prediction
    test_probs, y_true = get_probs(model, test_loader, device)
    y_pred = predict_with_threshold(test_probs, label_map)

    acc = (y_pred == y_true).mean()
    print(f"\n  Test accuracy (threshold): {acc:.4f}")
    print(f"  Per-class recall:")
    for ci in range(n_classes):
        mask = y_true == ci
        if mask.sum() == 0:
            continue
        recall = (y_pred[mask] == ci).mean()
        print(f"    {label_decode[ci]:15} recall={recall:.3f}  n={mask.sum()}")

    return y_true, y_pred, test_probs


def main():
    dataset_dir = (
        sys.argv[sys.argv.index("-dataset_dir") + 1]
        if "-dataset_dir" in sys.argv
        else "Dataset"
    )
    out_dir = (
        sys.argv[sys.argv.index("-out_dir") + 1]
        if "-out_dir" in sys.argv
        else "outputs"
    )

    models_dir = os.path.join(out_dir, "models")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(os.path.join(dataset_dir, "breathing_dataset.csv"))
    df = df[df["label"].isin(["Normal", "Hypopnea", "Apnea"])].reset_index(drop=True)

    labels = sorted(df["label"].unique())
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    label_decode = {i: lbl for lbl, i in label_map.items()}
    n_classes = len(labels)
    participants = sorted(df["participant_id"].unique())

    print(f"Classes: {label_map}")
    print(f"Participants: {participants}")
    print(f"Labels:\n{df['label'].value_counts().to_string()}")
    print(f"\nStarting LOPO — {len(participants)} folds")

    all_true, all_pred, all_probs = [], [], []

    for i, test_pid in enumerate(participants):
        train_df = df[df["participant_id"] != test_pid].reset_index(drop=True)
        test_df = df[df["participant_id"] == test_pid].reset_index(drop=True)

        y_true, y_pred, probs = train_fold(
            i + 1, test_pid, train_df, test_df, label_map, n_classes, models_dir, device
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        all_probs.extend(probs.tolist())

        fold_df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_true_label": [label_decode[v] for v in y_true],
                "y_pred_label": [label_decode[v] for v in y_pred],
                **{f"prob_{label_decode[ci]}": probs[:, ci] for ci in range(n_classes)},
            }
        )
        fold_df.to_csv(
            os.path.join(preds_dir, f"fold{i+1}_{test_pid}.csv"), index=False
        )

    agg_df = pd.DataFrame(
        {
            "y_true": all_true,
            "y_pred": all_pred,
            "y_true_label": [label_decode[v] for v in all_true],
            "y_pred_label": [label_decode[v] for v in all_pred],
            **{
                f"prob_{label_decode[ci]}": [p[ci] for p in all_probs]
                for ci in range(n_classes)
            },
        }
    )
    agg_df.to_csv(os.path.join(preds_dir, "all_predictions.csv"), index=False)

    print(f"\nAll predictions saved -> {preds_dir}")
    print("Run evaluate.py to see full metrics, confusion matrix and ROC curves.")


if __name__ == "__main__":
    main()
