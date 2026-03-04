import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
from collections import Counter


def find_file(folder, keyword, exclude=None):
    for filename in sorted(os.listdir(folder)):
        name_lower = filename.lower()
        if keyword.lower() not in name_lower:
            continue
        if exclude and any(ex.lower() in name_lower for ex in exclude):
            continue
        return os.path.join(folder, filename)
    raise FileNotFoundError(f"No file with '{keyword}' in {folder}")


def parse_ts(s):
    s = str(s).strip().replace(",", ".")
    ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if not pd.isna(ts):
        return ts.to_pydatetime()
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except:
            pass
    raise ValueError(f"Could not parse timestamp: {s}")


def simplify_event_type(event_type):
    et = event_type.lower()
    if "hypopnea" in et:
        return "Hypopnea"
    if "obstructive" in et or "central" in et or "mixed" in et or "apnea" in et:
        return "Apnea"
    return event_type


def load_signal(filepath):
    timestamps = []
    values = []
    reading = False

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "Data:":
                reading = True
                continue
            if not reading or not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                ts = parse_ts(parts[0])
                val = float(parts[1])
                timestamps.append(ts)
                values.append(val)
            except:
                continue

    series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
    return series.sort_index()


def load_events(filepath):
    events = []
    header_done = False

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                header_done = True
                continue
            if not header_done:
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            try:
                dt = parts[0].strip()
                date_str = dt.split(" ")[0]
                time_part = dt.split(" ")[1]
                start_str, end_str = time_part.split("-")

                start_ts = pd.Timestamp(parse_ts(date_str + " " + start_str))
                end_ts = pd.Timestamp(parse_ts(date_str + " " + end_str))

                if end_ts < start_ts:
                    end_ts += timedelta(days=1)

                events.append(
                    {
                        "start": start_ts,
                        "end": end_ts,
                        "event_type": simplify_event_type(parts[2].strip()),
                    }
                )
            except:
                continue

    return events


def infer_fs(series):
    diffs = np.diff(series.index.values.astype("datetime64[ns]"))
    if len(diffs) == 0:
        return 25
    dt = np.median(diffs).astype("timedelta64[ns]").astype(float) / 1e9
    if dt <= 0:
        return 25
    return round(1 / dt)


def bandpass_filter(values, fs):
    nyq = fs / 2.0
    b, a = butter(4, [0.17 / nyq, 0.4 / nyq], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(values) <= padlen:
        return values
    return filtfilt(b, a, values)


def normalize_airflow(values):
    # MAD normalization: robust to outliers unlike z-score
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    return (values - med) / mad


def scale_resp(values):
    return values / (np.std(values) + 1e-6)


def smooth_spo2(series):
    # rolling mean to smooth the steppy 4Hz SpO2 signal
    return series.rolling(window=5, center=True, min_periods=1).mean()


def get_event_color(event_type):
    et = event_type.lower()
    if "hypopnea" in et:
        return "#FFD700"
    if "apnea" in et:
        return "#FF4444"
    return "#FF8800"


def plot_window(flow, thorac, spo2, events, ws, we, title, pdf):
    win_dur = (we - ws).total_seconds()

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)

    for sig, ax, ylabel, color, lw in [
        (flow, ax1, "Nasal Flow (L/min)", "#1565C0", 0.9),
        (thorac, ax2, "Resp. Amplitude", "#E65100", 0.9),
        (spo2, ax3, "SpO2 (%)", "#212121", 1.2),
    ]:
        mask = (sig.index >= ws) & (sig.index < we)
        chunk = sig[mask]
        if len(chunk) == 0:
            continue
        x = (chunk.index - ws).total_seconds().values
        ax.plot(x, chunk.values, color=color, linewidth=lw, zorder=3)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
        ax.set_xlim(0, win_dur)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#aaa", linewidth=0.4, zorder=1)

    # clip flow axis to 99th percentile so outliers don't squash the signal
    mask_fl = (flow.index >= ws) & (flow.index < we)
    fl_vals = flow[mask_fl].values
    if len(fl_vals) > 0:
        lo = np.percentile(fl_vals, 1)
        hi = np.percentile(fl_vals, 99)
        pad = max(abs(hi - lo) * 0.1, 0.1)
        ax1.set_ylim(lo - pad, hi + pad)

    # zoom spo2 y-axis so dips are clearly visible
    mask_sp = (spo2.index >= ws) & (spo2.index < we)
    sp_vals = spo2[mask_sp].values
    if len(sp_vals) > 0:
        lo = max(80, np.nanmin(sp_vals) - 2)
        hi = min(100, np.nanmax(sp_vals) + 2)
        ax3.set_ylim(lo, hi)

    # overlay events as shaded bands with labels
    win_events = [ev for ev in events if ev["end"] >= ws and ev["start"] < we]

    for ev in win_events:
        ev_s = max(0, (ev["start"] - ws).total_seconds())
        ev_e = min(win_dur, (ev["end"] - ws).total_seconds())
        color = get_event_color(ev["event_type"])

        for ax in [ax1, ax2, ax3]:
            ax.axvspan(ev_s, ev_e, color=color, alpha=0.45, zorder=2, linewidth=0)

        mid = (ev_s + ev_e) / 2
        y_top = ax1.get_ylim()[1]
        ax1.text(
            mid,
            y_top * 0.92,
            ev["event_type"],
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="#222",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85, edgecolor="none"),
            zorder=5,
        )

    # x-axis timestamps on bottom plot only
    xticks = np.linspace(0, win_dur, 9)
    xlabels = [(ws + pd.Timedelta(seconds=s)).strftime("%H:%M:%S") for s in xticks]

    for ax in [ax1, ax2]:
        ax.set_xticks(xticks)
        ax.set_xticklabels([])

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6.5)
    ax3.set_xlabel("Time", fontsize=8)

    # legend on each subplot
    ax1.plot([], [], color="#1565C0", linewidth=1.2, label="Nasal Flow")
    ax2.plot([], [], color="#E65100", linewidth=1.2, label="Thoracic/Abdominal Resp.")
    ax3.plot([], [], color="#212121", linewidth=1.2, label="SpO2")
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if "-name" in sys.argv:
        folder = sys.argv[sys.argv.index("-name") + 1]
    else:
        print("Usage: python vis.py -name Data/AP01")
        sys.exit(1)

    participant_id = os.path.basename(folder.rstrip("/"))
    print(f"Loading {participant_id}...")

    flow = load_signal(find_file(folder, "flow", exclude=["events", "profile"]))
    thorac = load_signal(find_file(folder, "thorac", exclude=["events"]))
    spo2 = smooth_spo2(load_signal(find_file(folder, "spo2", exclude=["events"])))
    events = load_events(find_file(folder, "events", exclude=["sleep", "profile"]))

    print(f"  Flow:   {len(flow):,} samples")
    print(f"  Thorac: {len(thorac):,} samples")
    print(f"  SpO2:   {len(spo2):,} samples")
    print(f"  Events: {len(events)}")

    counts = Counter(ev["event_type"] for ev in events)
    for k, v in counts.items():
        print(f"    {k}: {v}")

    fs = infer_fs(flow)
    print(f"  Sampling rate: {fs} Hz")

    flow_f = pd.Series(
        normalize_airflow(bandpass_filter(flow.values, fs)), index=flow.index
    )
    thorac_f = pd.Series(
        scale_resp(bandpass_filter(thorac.values, fs)), index=thorac.index
    )

    os.makedirs("Visualizations", exist_ok=True)
    out_path = os.path.join("Visualizations", f"{participant_id}_visualization.pdf")

    t_start = min(flow.index[0], spo2.index[0])
    t_end = max(flow.index[-1], spo2.index[-1])

    windows = []
    t = t_start
    while t < t_end:
        windows.append((t, min(t + pd.Timedelta(minutes=5), t_end)))
        t += pd.Timedelta(minutes=5)

    print(f"  Writing {len(windows)} windows to PDF...")

    legend_patches = [
        mpatches.Patch(color="#FFD700", alpha=0.7, label="Hypopnea"),
        mpatches.Patch(color="#FF4444", alpha=0.7, label="Apnea"),
    ]

    with PdfPages(out_path) as pdf:

        # cover page
        fig_c, ax_c = plt.subplots(figsize=(16, 4))
        ax_c.axis("off")
        ax_c.text(
            0.5,
            0.72,
            f"Sleep Study — {participant_id}",
            ha="center",
            fontsize=20,
            fontweight="bold",
            transform=ax_c.transAxes,
        )
        ax_c.text(
            0.5,
            0.50,
            f"Recording: {t_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{t_end.strftime('%Y-%m-%d %H:%M')}   |   "
            f"Total Events: {len(events)}   |   "
            + "   ".join(f"{k}: {v}" for k, v in counts.items()),
            ha="center",
            fontsize=11,
            color="#444",
            transform=ax_c.transAxes,
        )
        ax_c.legend(
            handles=legend_patches,
            loc="lower center",
            fontsize=11,
            ncol=2,
            bbox_to_anchor=(0.5, 0.05),
            title="Event Types",
            title_fontsize=11,
        )
        pdf.savefig(fig_c, dpi=150, bbox_inches="tight")
        plt.close(fig_c)

        for i, (ws, we) in enumerate(windows):
            title = (
                f"{participant_id}  |  "
                f"{ws.strftime('%Y-%m-%d %H:%M')} to "
                f"{we.strftime('%Y-%m-%d %H:%M')}"
            )
            plot_window(flow_f, thorac_f, spo2, events, ws, we, title, pdf)

            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(windows)} pages")

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
