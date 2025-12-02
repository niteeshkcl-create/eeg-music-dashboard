import os
import numpy as np
import pandas as pd
import mne
from scipy import signal, stats

# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = os.path.dirname(__file__)
bids_root = os.path.join(BASE_DIR, "ds002725")

RESULTS_DIR = os.path.join(BASE_DIR, "precomputed_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_OPTIONS = {
    "classical": "classicalMusic",
    "gen1": "genMusic01",
    "gen2": "genMusic02",
    "gen3": "genMusic03",
}
TASK_LABELS = ["Classical Music", "Generative Music 1", "Generative Music 2", "Generative Music 3"]

FREQUENCY_BANDS_SUBJECT = {
    "Delta": {"range": (1, 4), "label": "Sleep/Deep Relaxation", "cmap": "Blues"},
    "Alpha": {"range": (8, 13), "label": "Calm/Relaxation", "cmap": "Greens"},
    "Beta": {"range": (13, 30), "label": "Anxiety/Alertness", "cmap": "Reds"},
}

FREQUENCY_BANDS_GENDER = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
}

TIME_WINDOW_SUBJECT = (0.0, 30.0)
TIME_WINDOW_GENDER = (0.0, 5.0)

THETA_TIME_WINDOWS = {
    "5s Before Onset": (-5.0, 0.0),
    "At Onset": (-0.5, 0.5),
    "5s After Onset": (0.0, 5.0),
}
THETA_BAND = (4, 8)

FRONTAL_PAIRS = {
    "F4-F3": ("F4", "F3"),
    "F8-F7": ("F8", "F7"),
    "AF4-AF3": ("AF4", "AF3"),
}
WINDOW_LENGTH = 2.0
WINDOW_STEP = 0.5
ANALYSIS_DURATION = 60.0

# ==============================================================
# SHARED HELPERS
# ==============================================================

EXCLUDE_CHANNELS = [
    "ECG",
    "ft_valance",
    "ft_arousal",
    "ft_x",
    "ft_y",
    "ft_ghostvalence",
    "ft_ghostarousal",
    "music",
    "trialtype",
    "sams_valence",
    "sams_arousal",
    "sams_valencert",
    "sams_arousalrt",
    "nback_stimuli",
    "nback_keypress",
]


def find_subjects(root):
    return sorted([d.replace("sub-", "") for d in os.listdir(root) if d.startswith("sub-")])


def load_preprocessed(root, subject, task):
    path = os.path.join(root, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_preproc_raw.fif")
    if os.path.exists(path):
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    return None


def load_events(root, subject, task):
    path = os.path.join(root, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_events.tsv")
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    return None


def find_music_onset(events_df, raw, analysis_duration=5.0):
    if events_df is None or len(events_df) == 0:
        total_duration = raw.times[-1]
        return max(analysis_duration + 5, total_duration * 0.15)

    onset_time = None
    if "onset" in events_df.columns:
        onset_time = events_df["onset"].iloc[0]
    elif "sample" in events_df.columns:
        onset_time = events_df["sample"].iloc[0] / raw.info["sfreq"]
    elif "latency" in events_df.columns:
        onset_time = events_df["latency"].iloc[0]

    total_duration = raw.times[-1]
    if (
        onset_time is None
        or onset_time < (analysis_duration + 5)
        or onset_time > (total_duration - analysis_duration - 5)
    ):
        onset_time = max(analysis_duration + 5, total_duration * 0.2)
    return onset_time


def compute_band_power(raw, tmin, tmax, fmin, fmax):
    raw_cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    n_samples = len(raw_cropped.times)
    if n_samples >= 2048:
        n_fft = 2048
    elif n_samples >= 1024:
        n_fft = 1024
    else:
        n_fft = min(512, n_samples)
    n_overlap = n_fft // 2
    spectrum = raw_cropped.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, verbose=False
    )
    psds = spectrum.get_data()
    return np.mean(psds, axis=1)

# ==============================================================
# TAB 1 – Subject topomaps: save per-subject band powers
# ==============================================================


def compute_subject_band_powers(root, subject, task, time_window, frequency_bands):
    raw = load_preprocessed(root, subject, task)
    if raw is None:
        return None, None, False

    events_df = load_events(root, subject, task)
    onset_time = find_music_onset(events_df, raw)

    eeg_channels = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
    raw_eeg = raw.copy().pick_channels(eeg_channels)

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_eeg.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    abs_tmin = onset_time + time_window[0]
    abs_tmax = onset_time + time_window[1]
    if abs_tmin < 0 or abs_tmax > raw_eeg.times[-1]:
        return None, None, False

    band_powers = {}
    for band_name, band_info in frequency_bands.items():
        try:
            band_powers[band_name] = compute_band_power(
                raw_eeg, abs_tmin, abs_tmax, *band_info["range"]
            )
        except Exception:
            return None, None, False

    return band_powers, raw_eeg.info, True


def save_tab1_subject_band_powers():
    print("TAB 1: Computing subject band powers...")
    subjects = find_subjects(bids_root)
    records = []

    for subject in subjects:
        print(f"  Subject {subject}")
        for music_key, task in TASK_OPTIONS.items():
            print(f"    Task {task}")
            band_powers, info, ok = compute_subject_band_powers(
                bids_root, subject, task, TIME_WINDOW_SUBJECT, FREQUENCY_BANDS_SUBJECT
            )
            if not ok:
                print("      Skipped (no data / window issue)")
                continue
            ch_names = info["ch_names"]
            for band, values in band_powers.items():
                for ch, val in zip(ch_names, values):
                    records.append(
                        {
                            "subject": subject,
                            "task_key": music_key,
                            "task_bids": task,
                            "band": band,
                            "channel": ch,
                            "power": float(val),
                        }
                    )

    df = pd.DataFrame(records)
    path = os.path.join(RESULTS_DIR, "tab1_subject_band_powers.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ==============================================================
# TAB 2 – Group alpha evolution: save aggregated powers
# ==============================================================


def collect_alpha_power_time_course(root, subjects, tasks, time_windows):
    task_time_powers = {task: {win: [] for win in time_windows} for task in tasks}
    info = None

    for task in tasks:
        for subject in subjects:
            raw = load_preprocessed(root, subject, task)
            if raw is None:
                continue
            eeg_channels = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
            raw_eeg = raw.copy().pick_channels(eeg_channels)
            try:
                montage = mne.channels.make_standard_montage("standard_1020")
                raw_eeg.set_montage(montage, on_missing="ignore")
            except Exception:
                pass
            if info is None:
                info = raw_eeg.info
            events_df = load_events(root, subject, task)
            onset_time = find_music_onset(events_df, raw)
            for win_name, (tmin, tmax) in time_windows.items():
                abs_tmin = onset_time + tmin
                abs_tmax = onset_time + tmax
                if abs_tmin < 0 or abs_tmax > raw_eeg.times[-1]:
                    continue
                alpha_power = compute_band_power(raw_eeg, abs_tmin, abs_tmax, 8, 13)
                task_time_powers[task][win_name].append(alpha_power)

    return task_time_powers, info


def save_tab2_group_alpha_evolution():
    print("TAB 2: Computing group alpha evolution...")
    subjects = find_subjects(bids_root)
    tasks = list(TASK_OPTIONS.values())
    time_windows = {
        "Baseline": (-2.0, 0.0),
        "Onset": (0.0, 2.0),
        "2s Post": (2.0, 4.0),
        "5s Post": (5.0, 7.0),
    }

    task_time_powers, info = collect_alpha_power_time_course(
        bids_root, subjects, tasks, time_windows
    )
    if info is None:
        print("  No data found.")
        return

    # Use channel names from info, but guard for variable-length arrays
    ch_names = info["ch_names"]

    rows = []
    for task in tasks:
        for win_name in time_windows.keys():
            arr_list = task_time_powers[task][win_name]
            if not arr_list:
                continue

            # Keep only arrays that match the expected channel count
            valid = [a for a in arr_list if a.shape[0] == len(ch_names)]
            if not valid:
                continue

            arr = np.stack(valid, axis=0)  # subjects x channels  (all same length now)

            for ch_idx, ch in enumerate(ch_names):
                vals = arr[:, ch_idx]
                rows.append(
                    {
                        "task_bids": task,
                        "window": win_name,
                        "channel": ch,
                        "mean_power": float(np.mean(vals)),
                        "std_power": float(np.std(vals)),
                        "n_subjects": int(vals.shape[0]),
                    }
                )

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "tab2_group_alpha_evolution.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ==============================================================
# TAB 3 – Gender analysis: band powers + beta/alpha ratio
# ==============================================================


def load_participant_data(root):
    p = os.path.join(root, "participants.tsv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, sep="\t")
    gender_col = None
    for col in df.columns:
        if col.lower() in ("sex", "gender"):
            gender_col = col
            break
    if gender_col is None:
        return None
    df["gender"] = df[gender_col]
    return df


def collect_band_powers_by_gender(
    root, subjects, tasks, frequency_bands, time_window, participant_df, calc_ratio=True
):
    gender_data = {}
    gender_ratio = {} if calc_ratio else None

    for subject in subjects:
        subj_id = f"sub-{subject}"
        row = participant_df[participant_df["participant_id"] == subj_id]
        if len(row) == 0:
            continue
        gender = row["gender"].values[0]

        if gender not in gender_data:
            gender_data[gender] = {b: {t: [] for t in tasks} for b in frequency_bands.keys()}
            if calc_ratio:
                gender_ratio[gender] = {t: [] for t in tasks}

        for task in tasks:
            raw = load_preprocessed(root, subject, task)
            if raw is None:
                continue
            events_df = load_events(root, subject, task)
            onset_time = find_music_onset(events_df, raw)

            eeg_channels = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
            raw_eeg = raw.copy().pick_channels(eeg_channels)

            abs_tmin = onset_time + time_window[0]
            abs_tmax = onset_time + time_window[1]
            if abs_tmin < 0 or abs_tmax > raw_eeg.times[-1]:
                continue

            band_powers_dict = {}
            for band_name, (fmin, fmax) in frequency_bands.items():
                try:
                    bp = compute_band_power(raw_eeg, abs_tmin, abs_tmax, fmin, fmax)
                    mean_bp = float(np.mean(bp))
                    gender_data[gender][band_name][task].append(mean_bp)
                    band_powers_dict[band_name] = mean_bp
                except Exception:
                    continue

            if calc_ratio and "Beta" in band_powers_dict and "Alpha" in band_powers_dict:
                if band_powers_dict["Alpha"] > 0:
                    ratio = band_powers_dict["Beta"] / band_powers_dict["Alpha"]
                    gender_ratio[gender][task].append(float(ratio))

    return gender_data, gender_ratio


def save_tab3_gender_analysis():
    print("TAB 3: Computing gender band powers + beta/alpha ratios...")
    part_df = load_participant_data(bids_root)
    if part_df is None:
        print("  participants.tsv missing or no gender column; skipping Tab 3.")
        return

    subjects = find_subjects(bids_root)
    tasks = list(TASK_OPTIONS.values())
    gender_data, gender_ratio = collect_band_powers_by_gender(
        bids_root, subjects, tasks, FREQUENCY_BANDS_GENDER, TIME_WINDOW_GENDER, part_df, True
    )

    # band powers
    records = []
    for gender, band_dict in gender_data.items():
        for band, task_dict in band_dict.items():
            for task_idx, task in enumerate(tasks):
                vals = task_dict[task]
                for v in vals:
                    records.append(
                        {
                            "gender": gender,
                            "band": band,
                            "task_bids": task,
                            "task_label": TASK_LABELS[task_idx],
                            "power": float(v),
                        }
                    )
    df = pd.DataFrame(records)
    path = os.path.join(RESULTS_DIR, "tab3_gender_band_powers.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")

    # ratios
    ratio_records = []
    for gender, task_dict in gender_ratio.items():
        for task_idx, task in enumerate(tasks):
            vals = task_dict[task]
            for v in vals:
                ratio_records.append(
                    {
                        "gender": gender,
                        "task_bids": task,
                        "task_label": TASK_LABELS[task_idx],
                        "beta_alpha_ratio": float(v),
                    }
                )
    rdf = pd.DataFrame(ratio_records)
    path2 = os.path.join(RESULTS_DIR, "tab3_gender_beta_alpha_ratio.csv")
    rdf.to_csv(path2, index=False)
    print(f"  Saved {path2}")


# ==============================================================
# TAB 4 – Theta band time-course topomaps
# ==============================================================


def compute_theta_power_time_window(raw, tmin, tmax, fmin=4, fmax=8):
    return compute_band_power(raw, tmin, tmax, fmin, fmax)


def collect_theta_power_time_course(root, subjects, tasks, time_windows, theta_band):
    task_time_powers = {t: {w: [] for w in time_windows} for t in tasks}
    info = None

    for subject in subjects:
        for task in tasks:
            raw = load_preprocessed(root, subject, task)
            if raw is None:
                continue
            events_df = load_events(root, subject, task)
            onset_time = find_music_onset(events_df, raw)

            eeg_channels = [ch for ch in raw.ch_names if ch not in EXCLUDE_CHANNELS]
            raw_eeg = raw.copy().pick_channels(eeg_channels)

            if info is None:
                try:
                    montage = mne.channels.make_standard_montage("standard_1020")
                    raw_eeg.set_montage(montage, on_missing="ignore", verbose=False)
                except Exception:
                    pass
                info = raw_eeg.info

            for wname, (rtmin, rtmax) in time_windows.items():
                abs_tmin = onset_time + rtmin
                abs_tmax = onset_time + rtmax
                if abs_tmin < 0 or abs_tmax > raw_eeg.times[-1]:
                    continue
                try:
                    p = compute_theta_power_time_window(
                        raw_eeg, abs_tmin, abs_tmax, theta_band[0], theta_band[1]
                    )
                    task_time_powers[task][wname].append(p)
                except Exception:
                    continue

    return task_time_powers, info


def save_tab4_theta_timecourse():
    print("TAB 4: Computing theta band time-course...")
    subjects = find_subjects(bids_root)
    tasks = list(TASK_OPTIONS.values())

    ttp, info = collect_theta_power_time_course(
        bids_root, subjects, tasks, THETA_TIME_WINDOWS, THETA_BAND
    )
    if info is None:
        print("  No info; skipping.")
        return
    ch_names = info["ch_names"]

    rows = []
    for task in tasks:
        for wname in THETA_TIME_WINDOWS.keys():
            arr_list = ttp[task][wname]
            if not arr_list:
                continue
            arr = np.stack(arr_list, axis=0)
            avg = np.mean(arr, axis=0)
            for ch, val in zip(ch_names, avg):
                rows.append(
                    {
                        "task_bids": task,
                        "window": wname,
                        "channel": ch,
                        "theta_power": float(val),
                    }
                )

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "tab4_theta_timecourse.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")


# ==============================================================
# TAB 5 – Frontal alpha asymmetry
# ==============================================================


def compute_alpha_power_sliding_window(
    raw, channel, tmin, tmax, window_length, window_step, fmin=8, fmax=13
):
    sfreq = raw.info["sfreq"]
    w_samples = int(window_length * sfreq)
    s_samples = int(window_step * sfreq)
    ch_idx = raw.ch_names.index(channel)
    data = raw.get_data(picks=[ch_idx], tmin=tmin, tmax=tmax)[0]
    n_samples = len(data)
    n_windows = int((n_samples - w_samples) / s_samples) + 1
    powers, times = [], []

    for i in range(n_windows):
        start = i * s_samples
        end = start + w_samples
        if end > n_samples:
            break
        window_data = data[start:end]
        freqs, psd = signal.welch(
            window_data, fs=sfreq, nperseg=min(len(window_data), 256), noverlap=128
        )
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        alpha_power = float(np.mean(psd[idx]))
        powers.append(alpha_power)
        center_time = (start + w_samples / 2) / sfreq
        times.append(center_time)

    return np.array(times), np.array(powers)


def compute_frontal_asymmetry_timeseries(
    raw, pair_name, left_ch, right_ch, tmin, tmax, window_length, window_step, fmin=8, fmax=13
):
    if left_ch not in raw.ch_names or right_ch not in raw.ch_names:
        return None, None
    t_l, p_l = compute_alpha_power_sliding_window(
        raw, left_ch, tmin, tmax, window_length, window_step, fmin, fmax
    )
    t_r, p_r = compute_alpha_power_sliding_window(
        raw, right_ch, tmin, tmax, window_length, window_step, fmin, fmax
    )
    n = min(len(p_l), len(p_r))
    p_l = p_l[:n]
    p_r = p_r[:n]
    times = t_l[:n]
    eps = 1e-10
    asym = np.log(p_r + eps) - np.log(p_l + eps)
    return times, asym


def collect_frontal_asymmetry_all_subjects(
    root, subjects, task, frontal_pairs, window_length, window_step, analysis_duration, alpha_band
):
    data = {pair: [] for pair in frontal_pairs.keys()}

    for subject in subjects:
        raw = load_preprocessed(root, subject, task)
        if raw is None:
            continue
        events_df = load_events(root, subject, task)
        onset = find_music_onset(events_df, raw)
        tmin = onset
        tmax = onset + analysis_duration
        if tmax > raw.times[-1]:
            continue

        for pair_name, (right_ch, left_ch) in frontal_pairs.items():
            times, asym = compute_frontal_asymmetry_timeseries(
                raw,
                pair_name,
                left_ch,
                right_ch,
                tmin,
                tmax,
                window_length,
                window_step,
                alpha_band[0],
                alpha_band[1],
            )
            if asym is not None:
                data[pair_name].append(
                    {"subject": subject, "times": times, "asymmetry": asym}
                )

    return data


def save_tab5_frontal_asymmetry():
    print("TAB 5: Computing frontal alpha asymmetry...")
    subjects = find_subjects(bids_root)
    task = "classicalMusic"

    asym_data = collect_frontal_asymmetry_all_subjects(
        bids_root,
        subjects,
        task,
        FRONTAL_PAIRS,
        WINDOW_LENGTH,
        WINDOW_STEP,
        ANALYSIS_DURATION,
        (8, 13),
    )

    # Save long-form CSV (subject, pair, time, asymmetry)
    rows = []
    for pair_name, entries in asym_data.items():
        for entry in entries:
            subj = entry["subject"]
            for t, a in zip(entry["times"], entry["asymmetry"]):
                rows.append(
                    {
                        "subject": subj,
                        "pair": pair_name,
                        "time": float(t),
                        "asymmetry": float(a),
                    }
                )
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_timeseries.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")

    # Also save pair-level summary stats
    stats_rows = []
    for pair_name, entries in asym_data.items():
        if not entries:
            continue
        all_vals = np.concatenate([e["asymmetry"] for e in entries])
        mean = float(np.mean(all_vals))
        sd = float(np.std(all_vals))
        median = float(np.median(all_vals))
        rng = (float(np.min(all_vals)), float(np.max(all_vals)))
        pct_pos = float((all_vals > 0).sum() / len(all_vals) * 100.0)
        pct_neg = float((all_vals < 0).sum() / len(all_vals) * 100.0)
        t_stat, p_val = stats.ttest_1samp(all_vals, 0.0)

        stats_rows.append(
            {
                "pair": pair_name,
                "n_values": int(len(all_vals)),
                "mean": mean,
                "sd": sd,
                "median": median,
                "min": rng[0],
                "max": rng[1],
                "pct_positive": pct_pos,
                "pct_negative": pct_neg,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }
        )

    sdf = pd.DataFrame(stats_rows)
    path2 = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_summary.csv")
    sdf.to_csv(path2, index=False)
    print(f"  Saved {path2}")
    
# ==============================================================
# TAB 6 – Arousal/Valence Analysis: per-trial emotional metrics
# ==============================================================

def compute_valence_arousal_trial(raw, onset_time, sfreq_target=200, duration=10.0):
    """
    Compute valence (F4-F3 alpha asymmetry) and arousal (beta/alpha ratio) for a trial.
    """
    filter_l, filter_h = 0.5, 50
    bands = {"alpha": (8, 13), "beta": (13, 30)}
    frontal_chs = ["F3", "F4"]
    arousal_chs = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "Cz"]
    
    try:
        # Crop to trial duration
        tmin = onset_time
        tmax = onset_time + duration
        raw_trial = raw.copy().crop(tmin=tmin, tmax=tmax)
        
        # Basic preprocessing (resample, filter)
        raw_trial.resample(sfreq_target, verbose=False)
        raw_trial.filter(filter_l, filter_h, verbose=False)
        
        # Simple ICA for EOG removal (using first 15 components)
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
        ica.fit(raw_trial)
        eog_indices, _ = ica.find_bads_eog(raw_trial, ch_name=["Fp1", "Fp2"])
        ica.exclude = eog_indices
        raw_clean = ica.apply(raw_trial.copy())
        
        # Get trial data
        start_sample = 0
        end_sample = int(duration * sfreq_target)
        seg = raw_clean.get_data(start=start_sample, stop=end_sample)
        
        # Valence: F4-F3 alpha asymmetry
        if "F3" in raw_clean.ch_names and "F4" in raw_clean.ch_names:
            f3_idx = raw_clean.ch_names.index("F3")
            f4_idx = raw_clean.ch_names.index("F4")
            alpha_f3 = bandpower(seg[f3_idx], sfreq_target, bands["alpha"])
            alpha_f4 = bandpower(seg[f4_idx], sfreq_target, bands["alpha"])
            valence = alpha_f4 - alpha_f3
        else:
            valence = np.nan
        
        # Arousal: mean beta/alpha ratio across frontal/central channels
        arousal_powers = []
        valid_arousal_chs = [ch for ch in arousal_chs if ch in raw_clean.ch_names]
        if valid_arousal_chs:
            for ch in valid_arousal_chs:
                ch_idx = raw_clean.ch_names.index(ch)
                beta_pow = bandpower(seg[ch_idx], sfreq_target, bands["beta"])
                alpha_pow = bandpower(seg[ch_idx], sfreq_target, bands["alpha"])
                if alpha_pow > 0:
                    arousal_powers.append(beta_pow / alpha_pow)
            
            arousal = np.mean(arousal_powers) if arousal_powers else np.nan
        else:
            arousal = np.nan
            
        return valence, arousal
        
    except Exception:
        return np.nan, np.nan


def bandpower(data, sf, band):
    """Helper: compute band power using Welch PSD."""
    f, Pxx = welch(data, sf, nperseg=min(sf*2, len(data)//2))
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.mean(Pxx[idx_band]) if np.any(idx_band) else 0.0


def find_music_trial_onsets(events_df):
    """Find music onset events (trial_type == 768 or first onset)."""
    if events_df is None or len(events_df) == 0:
        return []
    
    trial_onsets = []
    if "trial_type" in events_df.columns:
        music_rows = events_df[events_df["trial_type"] == 768]
        if len(music_rows) > 0:
            trial_onsets = music_rows["onset"].tolist()
    elif "onset" in events_df.columns:
        trial_onsets = events_df["onset"].tolist()[:5]  # Take first few if no trial_type
    
    return trial_onsets


def save_tab6_arousal_valence():
    print("TAB 6: Computing per-trial arousal/valence metrics...")
    subjects = find_subjects(bids_root)
    tasks = list(TASK_OPTIONS.values())
    
    records = []
    
    for subject in subjects:
        print(f"  Subject {subject}")
        for task_idx, task in enumerate(tasks):
            print(f"    Task {task}")
            
            # Try preprocessed FIF first, fallback to EDF
            raw_path_fif = os.path.join(bids_root, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_preproc_raw.fif")
            raw_path_edf = os.path.join(bids_root, f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_eeg.edf")
            
            raw = None
            if os.path.exists(raw_path_fif):
                raw = load_preprocessed(bids_root, subject, task)
            elif os.path.exists(raw_path_edf):
                raw = mne.io.read_raw_edf(raw_path_edf, preload=True, verbose=False)
            
            if raw is None:
                print(f"      No EEG data found")
                continue
            
            events_df = load_events(bids_root, subject, task)
            music_onsets = find_music_trial_onsets(events_df)
            
            if not music_onsets:
                # Fallback: use first onset or estimated onset
                first_onset = find_music_onset(events_df, raw, analysis_duration=10.0)
                music_onsets = [first_onset]
            
            # Compute valence/arousal for each trial
            for trial_idx, onset_time in enumerate(music_onsets, start=1):
                if onset_time + 10.0 > raw.times[-1]:
                    continue
                    
                valence, arousal = compute_valence_arousal_trial(raw, onset_time)
                
                records.append({
                    "subject": subject,
                    "task_key": list(TASK_OPTIONS.keys())[task_idx],
                    "task_bids": task,
                    "task_label": TASK_LABELS[task_idx],
                    "trial": trial_idx,
                    "music_onset": float(onset_time),
                    "valence": float(valence),
                    "arousal": float(arousal),
                })
    
    df = pd.DataFrame(records)
    path = os.path.join(RESULTS_DIR, "tab6_arousal_valence_trials.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} records to {path}")
    
    # Also save summary statistics per subject/task
    summary_records = []
    for (subject, task_key), group in df.groupby(["subject", "task_key"]):
        summary_records.append({
            "subject": subject,
            "task_key": task_key,
            "n_trials": len(group),
            "valence_mean": float(group["valence"].mean()),
            "valence_std": float(group["valence"].std()),
            "arousal_mean": float(group["arousal"].mean()),
            "arousal_std": float(group["arousal"].std()),
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(RESULTS_DIR, "tab6_arousal_valence_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved subject/task summary to {summary_path}")



# ==============================================================
# MAIN
# ==============================================================


if __name__ == "__main__":
    print("Precomputing all tab outputs from full ds002725 ...")
    save_tab1_subject_band_powers()
    save_tab2_group_alpha_evolution()
    save_tab3_gender_analysis()
    save_tab4_theta_timecourse()
    save_tab5_frontal_asymmetry()
    save_tab6_arousal_valence()  # <-- NEW
    print("Done. All outputs written to precomputed_results/")

