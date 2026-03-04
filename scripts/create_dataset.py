import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt


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
    raise ValueError(f"Could not parse: {s}")


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
                val = float(parts[1].strip())
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
                date_and_times = parts[0].strip()
                date_str = date_and_times[:10]
                time_part = date_and_times[11:]
                start_str, end_str = time_part.split("-")

                start_ts = pd.Timestamp(parse_ts(date_str + " " + start_str))
                end_ts = pd.Timestamp(parse_ts(date_str + " " + end_str))

                if end_ts < start_ts:
                    end_ts = end_ts + timedelta(days=1)

                events.append(
                    {"start": start_ts, "end": end_ts, "event_type": parts[2].strip()}
                )
            except:
                continue
    return events


def load_sleep_profile(filepath):
    timestamps = []
    stages = []
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
            if len(parts) < 2:
                continue
            try:
                ts = parse_ts(parts[0])
                stage = parts[1].strip()
                timestamps.append(ts)
                stages.append(stage)
            except:
                continue

    series = pd.Series(stages, index=pd.DatetimeIndex(timestamps))
    return series.sort_index()


def bandpass_filter(values, fs):
    nyq = fs / 2.0
    b, a = butter(4, [0.17 / nyq, 0.4 / nyq], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(values) <= padlen:
        return values
    return filtfilt(b, a, values)


def normalize(values):
    mu = np.mean(values)
    std = np.std(values)
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mu) / std


def simplify_label(event_type):
    et = event_type.strip().lower()
    if "hypopnea" in et:
        return "Hypopnea"
    if "obstructive" in et or "central" in et or "mixed" in et or "apnea" in et:
        return "Apnea"
    # Body event, artifact, or anything else -> Normal
    return "Normal"


def get_label(win_start, win_end, events):
    win_dur = (win_end - win_start).total_seconds()
    best_label = "Normal"
    best_frac = 0.0

    for ev in events:
        overlap_start = max(win_start, ev["start"])
        overlap_end = min(win_end, ev["end"])
        overlap_sec = (overlap_end - overlap_start).total_seconds()
        if overlap_sec <= 0:
            continue
        frac = overlap_sec / win_dur
        if frac > 0.5 and frac > best_frac:
            best_frac = frac
            best_label = simplify_label(ev["event_type"])

    return best_label


def get_sleep_stage(win_start, win_end, sleep_profile):
    mask = (sleep_profile.index >= win_start) & (sleep_profile.index <= win_end)
    chunk = sleep_profile[mask]
    if len(chunk) == 0:
        before = sleep_profile[sleep_profile.index <= win_start]
        return before.iloc[-1] if len(before) > 0 else "Unknown"
    return chunk.mode()[0]


def resample_spo2(spo2, flow):
    merged = spo2.reindex(spo2.index.union(flow.index)).ffill().bfill()
    return merged.reindex(flow.index)


def process_participant(folder):
    pid = os.path.basename(folder)
    print(f"\nProcessing {pid}...")

    flow = load_signal(find_file(folder, "flow", exclude=["events", "profile"]))
    thorac = load_signal(find_file(folder, "thorac", exclude=["events"]))
    spo2 = load_signal(find_file(folder, "spo2", exclude=["events"]))
    events = load_events(find_file(folder, "events", exclude=["sleep", "profile"]))
    sleep_profile = load_sleep_profile(find_file(folder, "sleep", exclude=["events"]))

    flow_f = pd.Series(bandpass_filter(flow.values, 32), index=flow.index)
    thorac_f = pd.Series(bandpass_filter(thorac.values, 32), index=thorac.index)
    spo2_32 = resample_spo2(spo2, flow_f)

    n_samples = 30 * 32  # 960 samples per window
    step = 15 * 32  # 50% overlap

    breathing_rows = []
    sleep_stage_rows = []
    i = 0

    while i + n_samples <= len(flow_f):
        win_flow = normalize(flow_f.values[i : i + n_samples])
        win_thorac = normalize(thorac_f.values[i : i + n_samples])
        win_spo2 = normalize(spo2_32.values[i : i + n_samples])

        win_start = flow_f.index[i]
        win_end = flow_f.index[i + n_samples - 1]

        label = get_label(win_start, win_end, events)
        stage = get_sleep_stage(win_start, win_end, sleep_profile)

        row = {
            "participant_id": pid,
            "window_index": len(breathing_rows),
            "start_time": win_start,
            "end_time": win_end,
            "label": label,
            "sleep_stage": stage,
        }
        for j in range(n_samples):
            row[f"flow_{j}"] = win_flow[j]
            row[f"thorac_{j}"] = win_thorac[j]
            row[f"spo2_{j}"] = win_spo2[j]

        breathing_rows.append(row)
        sleep_stage_rows.append(
            {
                "participant_id": pid,
                "window_index": len(sleep_stage_rows),
                "start_time": win_start,
                "end_time": win_end,
                "label": label,
                "sleep_stage": stage,
            }
        )

        i += step

    print(f"  Windows: {len(breathing_rows)}")
    print(
        f"  Labels:\n{pd.Series([r['label'] for r in breathing_rows]).value_counts().to_string()}"
    )

    return breathing_rows, sleep_stage_rows


def main():
    in_dir = (
        sys.argv[sys.argv.index("-in_dir") + 1] if "-in_dir" in sys.argv else "Data"
    )
    out_dir = (
        sys.argv[sys.argv.index("-out_dir") + 1]
        if "-out_dir" in sys.argv
        else "Dataset"
    )

    os.makedirs(out_dir, exist_ok=True)

    folders = []
    for name in sorted(os.listdir(in_dir)):
        full = os.path.join(in_dir, name)
        if os.path.isdir(full) and not name.startswith("."):
            folders.append(full)

    print(
        f"Found {len(folders)} participant(s): {[os.path.basename(f) for f in folders]}"
    )

    all_breathing, all_sleep_stage = [], []

    for folder in folders:
        b_rows, s_rows = process_participant(folder)
        all_breathing.extend(b_rows)
        all_sleep_stage.extend(s_rows)

    df_breathing = pd.DataFrame(all_breathing)
    df_sleep = pd.DataFrame(all_sleep_stage)

    breathing_path = os.path.join(out_dir, "breathing_dataset.csv")
    sleep_stage_path = os.path.join(out_dir, "sleep_stage_dataset.csv")

    df_breathing.to_csv(breathing_path, index=False)
    df_sleep.to_csv(sleep_stage_path, index=False)

    print(f"\nSaved -> {breathing_path}   ({df_breathing.shape[0]} rows)")
    print(f"Saved -> {sleep_stage_path}  ({df_sleep.shape[0]} rows)")
    print(f"\nOverall label distribution:")
    print(df_sleep["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
