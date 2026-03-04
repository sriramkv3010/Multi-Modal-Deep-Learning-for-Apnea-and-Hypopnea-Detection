import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, classes):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0, labels=classes
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0, labels=classes
            ),
            4,
        ),
        "f1": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=classes),
            4,
        ),
    }


def plot_confusion_matrix(ax, cm, classes, title):
    cm_norm = cm.astype(float) / np.where(
        cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True)
    )
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(classes)):
        for j in range(len(classes)):
            annot[i, j] = f"{cm_norm[i,j]:.2f}\n({cm[i,j]})"

    sns.heatmap(
        cm_norm,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)


def plot_roc_curves(ax, y_true, probs_df, classes, title):
    colors = ["#FF4444", "#FFD700", "#1565C0"]
    y_bin = label_binarize(y_true, classes=classes)

    for i, cls in enumerate(classes):
        col = f"prob_{cls}"
        if col not in probs_df.columns:
            continue
        probs = probs_df[col].values

        if y_bin.shape[1] == 1:
            fpr, tpr, _ = roc_curve(y_bin[:, 0], probs)
        else:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs)

        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(
            fpr, tpr, color=color, linewidth=2, label=f"{cls}  (AUC = {roc_auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def plot_per_class_recall(ax, fold_names, fold_metrics_per_class, classes):
    x = np.arange(len(fold_names))
    width = 0.25
    colors = ["#FF4444", "#FFD700", "#1565C0"]

    for i, cls in enumerate(classes):
        recalls = [m.get(cls, 0) for m in fold_metrics_per_class]
        ax.bar(
            x + i * width,
            recalls,
            width,
            label=cls,
            color=colors[i % len(colors)],
            edgecolor="white",
            alpha=0.85,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(fold_names, rotation=30, fontsize=8)
    ax.set_ylabel("Recall", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class Recall per Fold", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    pred_dir = (
        sys.argv[sys.argv.index("-pred_dir") + 1]
        if "-pred_dir" in sys.argv
        else "outputs/predictions"
    )
    out_dir = (
        sys.argv[sys.argv.index("-out_dir") + 1]
        if "-out_dir" in sys.argv
        else "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    all_df = pd.read_csv(os.path.join(pred_dir, "all_predictions.csv"))
    classes = sorted(all_df["y_true_label"].unique())

    fold_files = sorted(
        [f for f in os.listdir(pred_dir) if f.startswith("fold") and f.endswith(".csv")]
    )

    fold_names = []
    fold_metrics = []
    fold_metrics_per_class = []

    print(f"\n{'Fold':<22} {'Acc':>7} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 58)

    for fname in fold_files:
        df = pd.read_csv(os.path.join(pred_dir, fname))
        m = compute_metrics(df["y_true_label"], df["y_pred_label"], classes)
        name = fname.replace(".csv", "")
        fold_names.append(name)
        fold_metrics.append(m)
        print(
            f"{name:<22} {m['accuracy']:>7.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f}"
        )

        # per-class recall for this fold
        per_cls = {}
        for cls in classes:
            mask = df["y_true_label"] == cls
            if mask.sum() == 0:
                per_cls[cls] = 0.0
            else:
                per_cls[cls] = (df.loc[mask, "y_pred_label"] == cls).mean()
        fold_metrics_per_class.append(per_cls)

    overall = compute_metrics(all_df["y_true_label"], all_df["y_pred_label"], classes)
    cm_all = confusion_matrix(
        all_df["y_true_label"], all_df["y_pred_label"], labels=classes
    )

    print("-" * 58)
    print(
        f"{'OVERALL':<22} {overall['accuracy']:>7.4f} {overall['precision']:>8.4f} "
        f"{overall['recall']:>8.4f} {overall['f1']:>8.4f}"
    )

    print("\n--- Per-Class Recall ---")
    for cls in classes:
        mask = all_df["y_true_label"] == cls
        recall = (all_df.loc[mask, "y_pred_label"] == cls).mean()
        n = mask.sum()
        print(f"  {cls:20} recall={recall:.3f}  n={n}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            all_df["y_true_label"],
            all_df["y_pred_label"],
            labels=classes,
            zero_division=0,
        )
    )

    rows = [{"fold": n, **m} for n, m in zip(fold_names, fold_metrics)]
    rows.append({"fold": "OVERALL", **overall})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # ── PDF report ──
    pdf_path = os.path.join(out_dir, "evaluation_report.pdf")
    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Overall confusion matrix + ROC curves ──
        fig1, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))
        fig1.suptitle("Overall Results", fontsize=13, fontweight="bold")

        plot_confusion_matrix(ax_cm, cm_all, classes, "Confusion Matrix (All Folds)")
        plot_roc_curves(
            ax_roc,
            all_df["y_true_label"].values,
            all_df,
            classes,
            "ROC Curves (All Folds)",
        )

        plt.tight_layout()
        pdf.savefig(fig1, dpi=150, bbox_inches="tight")
        plt.close(fig1)

        # ── Page 2: Per-class recall bars + overall metrics bar ──
        fig2, (ax_recall, ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle("Per-Fold Analysis", fontsize=13, fontweight="bold")

        plot_per_class_recall(ax_recall, fold_names, fold_metrics_per_class, classes)

        metrics_list = ["accuracy", "precision", "recall", "f1"]
        vals = [overall[m] for m in metrics_list]
        colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
        bars = ax_metrics.bar(
            metrics_list, vals, color=colors, edgecolor="white", width=0.5
        )
        ax_metrics.set_ylim(0, 1.15)
        ax_metrics.set_title("Overall Metrics", fontsize=11, fontweight="bold")
        ax_metrics.grid(axis="y", linestyle="--", alpha=0.4)
        ax_metrics.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            ax_metrics.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        pdf.savefig(fig2, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # ── Page 3: Per-fold confusion matrices ──
        n = len(fold_files)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
        fig3.suptitle("Per-Fold Confusion Matrices", fontsize=13, fontweight="bold")
        axes3_flat = np.array(axes3).flatten()

        for idx, fname in enumerate(fold_files):
            df = pd.read_csv(os.path.join(pred_dir, fname))
            cm_fold = confusion_matrix(
                df["y_true_label"], df["y_pred_label"], labels=classes
            )
            plot_confusion_matrix(
                axes3_flat[idx], cm_fold, classes, fname.replace(".csv", "")
            )

        for idx in range(n, len(axes3_flat)):
            axes3_flat[idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig3, dpi=150, bbox_inches="tight")
        plt.close(fig3)

        # ── Page 4: Per-fold ROC curves ──
        fig4, axes4 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
        fig4.suptitle("Per-Fold ROC Curves", fontsize=13, fontweight="bold")
        axes4_flat = np.array(axes4).flatten()

        for idx, fname in enumerate(fold_files):
            df = pd.read_csv(os.path.join(pred_dir, fname))
            plot_roc_curves(
                axes4_flat[idx],
                df["y_true_label"].values,
                df,
                classes,
                fname.replace(".csv", ""),
            )

        for idx in range(n, len(axes4_flat)):
            axes4_flat[idx].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig4, dpi=150, bbox_inches="tight")
        plt.close(fig4)

    print(f"\nReport saved -> {pdf_path}")
    print(f"Metrics CSV -> {out_dir}/metrics_summary.csv")


if __name__ == "__main__":
    main()
