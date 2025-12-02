import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import streamlit as st

# ========================
# PATHS & CONFIG
# ========================

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "precomputed_results")

TASK_OPTIONS = {
    "classical": "classicalMusic",
    "gen1": "genMusic01",
    "gen2": "genMusic02",
    "gen3": "genMusic03",
}
TASK_LABELS = ["Classical Music", "Generative Music 1", "Generative Music 2", "Generative Music 3"]

FREQUENCY_BANDS_SUBJECT = {
    "Delta": {"label": "Sleep/Deep Relaxation", "cmap": "Blues"},
    "Alpha": {"label": "Calm/Relaxation", "cmap": "Greens"},
    "Beta": {"label": "Anxiety/Alertness", "cmap": "Reds"},
}

THETA_WINDOWS_ORDER = ["5s Before Onset", "At Onset", "5s After Onset"]
ALPHA_EV_WINDOWS_ORDER = ["Baseline", "Onset", "2s Post", "5s Post"]

# ========================
# LOAD PRECOMPUTED DATA
# ========================

@st.cache_data
def load_tab1_data():
    path = os.path.join(RESULTS_DIR, "tab1_subject_band_powers.csv")
    return pd.read_csv(path)

@st.cache_data
def load_tab2_data():
    path = os.path.join(RESULTS_DIR, "tab2_group_alpha_evolution.csv")
    return pd.read_csv(path)

@st.cache_data
def load_tab3_data():
    p1 = os.path.join(RESULTS_DIR, "tab3_gender_band_powers.csv")
    p2 = os.path.join(RESULTS_DIR, "tab3_gender_beta_alpha_ratio.csv")
    return pd.read_csv(p1), pd.read_csv(p2)

@st.cache_data
def load_tab4_data():
    path = os.path.join(RESULTS_DIR, "tab4_theta_timecourse.csv")
    return pd.read_csv(path)

@st.cache_data
def load_tab5_data():
    p1 = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_timeseries.csv")
    p2 = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_summary.csv")
    return pd.read_csv(p1), pd.read_csv(p2)

# Utility to build fake MNE Info from channel list (for topomaps)
def make_info_from_channels(ch_names, sfreq=250.0):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info

# ========================
# STREAMLIT UI
# ========================

st.set_page_config(page_title="EEG Music Analysis Dashboard", layout="wide")
st.title("🧠 EEG Music Analysis Dashboard")

# tab1, tab2, tab3, tab4, tab5 = st.tabs(
#     ["Individual Subject", "Group Evolution", "Gender Analysis", "Theta Time Course", "Frontal Asymmetry"]
# )

tab1, tab3, tab4, tab5 = st.tabs(
    ["Individual Subject", "Gender Analysis", "Theta Time Course", "Frontal Asymmetry"]
)

# =====================================================
# TAB 1 – Individual Subject Topomaps (from tab1 CSV)
# =====================================================

with tab1:
    st.header("Individual Subject Topomaps")

    df1 = load_tab1_data()

    subjects = sorted(df1["subject"].unique())
    task_keys = list(TASK_OPTIONS.keys())
    bands = list(FREQUENCY_BANDS_SUBJECT.keys())

    col1, col2 = st.columns(2)
    with col1:
        subj = st.selectbox("Subject", subjects)
    with col2:
        music_key = st.selectbox("Music Type", task_keys)

    band_sel = st.multiselect("Frequency bands", bands, default=bands)

    if st.button("Show Topomaps (Precomputed)", type="primary"):
        sub_df = df1[(df1["subject"] == subj) & (df1["task_key"] == music_key)]
        if sub_df.empty:
            st.error("No data for this subject / music combination.")
        else:
            ch_names = sorted(sub_df["channel"].unique())
            info = make_info_from_channels(ch_names)

            n_bands = len(band_sel)
            fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5))
            if n_bands == 1:
                axes = [axes]

            for ax, band in zip(axes, band_sel):
                band_df = sub_df[sub_df["band"] == band]
                band_df = band_df.set_index("channel").reindex(ch_names)
                vals = band_df["power"].to_numpy()

                mne.viz.plot_topomap(
                    vals,
                    info,
                    axes=ax,
                    show=False,
                    cmap=FREQUENCY_BANDS_SUBJECT[band]["cmap"],
                    contours=6,
                    extrapolate="head",
                    sphere=(0.0, 0.0, 0.09, 0.2),
                )
                ax.set_title(f"{band}\n{FREQUENCY_BANDS_SUBJECT[band]['label']}", fontsize=13)

            st.pyplot(fig)

# =====================================================
# # TAB 2 – Group Alpha Evolution (from tab2 CSV)
# # =====================================================

# with tab2:
#     st.header("Group Alpha Power Evolution")

#     df2 = load_tab2_data()
#     df2["task_label"] = df2["task_bids"].map(
#         {v: lbl for v, lbl in zip(TASK_OPTIONS.values(), TASK_LABELS)}
#     )

#     task_options = sorted(df2["task_label"].unique())
#     tasks_chosen = st.multiselect("Music Types", task_options, default=task_options)

#     if st.button("Show Group Evolution (Precomputed)", type="primary"):
#         plot_windows = ["Onset", "2s Post", "5s Post"]
#         base_win = "Baseline"

#         sub = df2[df2["task_label"].isin(tasks_chosen)]
#         if sub.empty:
#             st.error("No data for selected tasks.")
#         else:
#             # build channel list and info from any subset
#             ch_names = sorted(sub["channel"].unique())
#             info = make_info_from_channels(ch_names)

#             n_tasks = len(tasks_chosen)
#             n_times = len(plot_windows)
#             fig, axes = plt.subplots(n_tasks, n_times, figsize=(5 * n_times, 4 * n_tasks))
#             if n_tasks == 1:
#                 axes = np.array([axes])

#             # compute % change per task/window/channel
#             all_vals = []
#             pct_maps = {t: {} for t in tasks_chosen}
#             for t in tasks_chosen:
#                 tdf = sub[sub["task_label"] == t]
#                 base = tdf[tdf["window"] == base_win].set_index("channel")["mean_power"]
#                 for w in plot_windows:
#                     cur = tdf[tdf["window"] == w].set_index("channel")["mean_power"]
#                     cur = cur.reindex(base.index)
#                     base_vals = base.to_numpy()
#                     cur_vals = cur.to_numpy()
#                     with np.errstate(divide="ignore", invalid="ignore"):
#                         pct = (cur_vals - base_vals) / base_vals * 100.0
#                     pct_maps[t][w] = pct
#                     all_vals.append(pct)

#             if not all_vals:
#                 st.error("Insufficient baseline/current data to compute percent change.")
#             else:
#                 flat = np.concatenate(all_vals)
#                 limit = np.nanpercentile(np.abs(flat), 95)
#                 vmin, vmax = -limit, limit

#                 last_im = None
#                 for ti, t in enumerate(tasks_chosen):
#                     for wi, w in enumerate(plot_windows):
#                         ax = axes[ti, wi]
#                         if w not in pct_maps[t]:
#                             ax.axis("off")
#                             continue
#                         vals = pct_maps[t][w]
#                         im, _ = mne.viz.plot_topomap(
#                             vals,
#                             info,
#                             axes=ax,
#                             show=False,
#                             vlim=(vmin, vmax),
#                             cmap="RdBu_r",
#                             contours=5,
#                             sphere=(0.0, 0.0, 0.09, 0.2),
#                         )
#                         last_im = im
#                         if ti == 0:
#                             ax.set_title(w, fontsize=11, fontweight="bold")
#                         if wi == 0:
#                             ax.set_ylabel(t, fontsize=11, fontweight="bold")

#                 if last_im is not None:
#                     fig.subplots_adjust(right=0.90)
#                     cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
#                     cbar = fig.colorbar(last_im, cax=cax)
#                     cbar.set_label("Alpha % change vs. baseline", fontsize=11)

#                 st.pyplot(fig)

# =====================================================
# TAB 3 – Gender Analysis (from tab3 CSVs)
# =====================================================

with tab3:
    st.header("Gender-based Band Power & Beta/Alpha Ratio")

    band_df, ratio_df = load_tab3_data()

    if st.checkbox("Show band power plots", value=True):
        fig = plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(2, 2, 1)
        sns.barplot(
            data=band_df,
            x="band",
            y="power",
            hue="gender",
            ax=ax1,
            errorbar="se",
        )
        ax1.set_title("Mean Band Power by Gender")
        ax1.set_ylabel("Power (μV²/Hz)")

        ax2 = plt.subplot(2, 2, 2)
        sns.boxplot(
            data=band_df,
            x="band",
            y="power",
            hue="gender",
            ax=ax2,
        )
        ax2.set_title("Band Power Distribution")
        ax2.set_ylabel("Power (μV²/Hz)")

        ax3 = plt.subplot(2, 1, 2)
        sns.barplot(
            data=band_df,
            x="task_label",
            y="power",
            hue="gender",
            ax=ax3,
            errorbar="se",
        )
        ax3.set_title("Band Power by Task and Gender")
        ax3.set_ylabel("Power (μV²/Hz)")
        ax3.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    if st.checkbox("Show Beta/Alpha ratio plots", value=True):
        fig = plt.figure(figsize=(12, 6))

        ax1 = plt.subplot(1, 2, 1)
        sns.barplot(
            data=ratio_df,
            x="gender",
            y="beta_alpha_ratio",
            ax=ax1,
            errorbar="se",
        )
        ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax1.set_title("Beta/Alpha Ratio by Gender")
        ax1.set_ylabel("Beta/Alpha Ratio")

        ax2 = plt.subplot(1, 2, 2)
        sns.barplot(
            data=ratio_df,
            x="task_label",
            y="beta_alpha_ratio",
            hue="gender",
            ax=ax2,
            errorbar="se",
        )
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax2.set_title("Beta/Alpha Ratio by Task and Gender")
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig)

# =====================================================
# TAB 4 – Theta band time-course (from tab4 CSV)
# =====================================================

with tab4:
    st.header("Theta Band Time-Course Topomaps")

    df4 = load_tab4_data()
    df4["task_label"] = df4["task_bids"].map(
        {v: lbl for v, lbl in zip(TASK_OPTIONS.values(), TASK_LABELS)}
    )

    tasks_choice = st.multiselect(
        "Music Types", sorted(df4["task_label"].unique()), default=sorted(df4["task_label"].unique())
    )

    if st.button("Show Theta Topomaps (Precomputed)", type="primary"):
        sub = df4[df4["task_label"].isin(tasks_choice)]
        if sub.empty:
            st.error("No data for selected tasks.")
        else:
            ch_names = sorted(sub["channel"].unique())
            info = make_info_from_channels(ch_names)

            n_tasks = len(tasks_choice)
            n_times = len(THETA_WINDOWS_ORDER)
            fig, axes = plt.subplots(n_tasks, n_times, figsize=(5 * n_times, 4 * n_tasks))
            if n_tasks == 1:
                axes = np.array([axes])

            all_vals = []
            theta_maps = {t: {} for t in tasks_choice}
            for t in tasks_choice:
                tdf = sub[sub["task_label"] == t]
                for w in THETA_WINDOWS_ORDER:
                    wdf = tdf[tdf["window"] == w]
                    if wdf.empty:
                        continue
                    wdf = wdf.set_index("channel").reindex(ch_names)
                    vals = wdf["theta_power"].to_numpy()
                    theta_maps[t][w] = vals
                    all_vals.append(vals)

            if not all_vals:
                st.error("Not enough data for selected tasks/windows.")
            else:
                flat = np.concatenate(all_vals)
                vmin, vmax = np.nanpercentile(flat, [5, 95])

                last_im = None
                for ti, t in enumerate(tasks_choice):
                    for wi, w in enumerate(THETA_WINDOWS_ORDER):
                        ax = axes[ti, wi]
                        if w not in theta_maps[t]:
                            ax.axis("off")
                            continue
                        vals = theta_maps[t][w]
                        im, _ = mne.viz.plot_topomap(
                            vals,
                            info,
                            axes=ax,
                            show=False,
                            vlim=(vmin, vmax),
                            cmap="viridis",
                            contours=6,
                            extrapolate="head",
                            sphere="auto",
                        )
                        last_im = im
                        if ti == 0:
                            ax.set_title(w, fontsize=11, fontweight="bold")
                        if wi == 0:
                            ax.set_ylabel(t, fontsize=11, fontweight="bold")

                if last_im is not None:
                    fig.subplots_adjust(right=0.90)
                    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
                    cbar = fig.colorbar(last_im, cax=cax)
                    cbar.set_label("Theta Power (μV²/Hz)", fontsize=11)

                st.pyplot(fig)

# =====================================================
# TAB 5 – Frontal alpha asymmetry (from tab5 CSVs)
# =====================================================

with tab5:
    st.header("Frontal Alpha Asymmetry (Classical Music)")

    ts_df, summary_df = load_tab5_data()

    pairs = sorted(ts_df["pair"].unique())
    pair = st.selectbox("Electrode pair", pairs)

    pair_df = ts_df[ts_df["pair"] == pair]
    if pair_df.empty:
        st.error("No data for selected pair.")
    else:
        # heatmap matrix: subjects x time
        subjects = sorted(pair_df["subject"].unique())
        times = np.sort(pair_df["time"].unique())
        mat = np.zeros((len(subjects), len(times)))

        sub_to_row = {s: i for i, s in enumerate(subjects)}
        time_to_col = {t: j for j, t in enumerate(times)}

        for _, row in pair_df.iterrows():
            i = sub_to_row[row["subject"]]
            j = time_to_col[row["time"]]
            mat[i, j] = row["asymmetry"]

        fig = plt.figure(figsize=(14, 6))

        # heatmap
        ax1 = plt.subplot(1, 2, 1)
        vmax = np.percentile(np.abs(mat), 95)
        vmin = -vmax
        im = ax1.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[times[0], times[-1], len(subjects) - 0.5, -0.5],
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Subject index")
        ax1.set_title(f"{pair} Asymmetry (Right - Left)")
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Asymmetry")

        # mean curve
        ax2 = plt.subplot(1, 2, 2)
        mean_asym = np.mean(mat, axis=0)
        std_asym = np.std(mat, axis=0)
        sem = std_asym / np.sqrt(mat.shape[0])

        ax2.plot(times, mean_asym, "b-", label="Mean asymmetry")
        ax2.fill_between(
            times,
            mean_asym - 1.96 * sem,
            mean_asym + 1.96 * sem,
            alpha=0.3,
            color="blue",
            label="95% CI",
        )
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Asymmetry (R-L)")
        ax2.set_title(f"{pair} Mean Asymmetry Over Time")
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Summary statistics")
        st.dataframe(summary_df[summary_df["pair"] == pair])
