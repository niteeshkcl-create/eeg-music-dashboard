

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import mne
# import streamlit as st

# st.set_page_config(page_title="EEG Music Analysis Dashboard", layout="wide")

# st.markdown("""
#     <style>
#     /* Make whole page black */
#     html, body, [data-testid="stAppViewContainer"] {
#         background-color: #000000 !important;
#         color: #ffffff !important;
#     }

#     /* Main app block */
#     [data-testid="stAppViewContainer"] > .main {
#         background-color: #000000 !important;
#         color: #ffffff !important;
#     }

#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #000000 !important;
#         color: #ffffff !important;
#     }

#     /* Top header area */
#     [data-testid="stHeader"] {
#         background-color: #000000 !important;
#     }

#     /* Tabs bar */
#     .stTabs [data-baseweb="tab-list"] {
#         background-color: #000000 !important;
#     }

#     /* Tabs */
#     .stTabs [data-baseweb="tab"] {
#         background-color: #111111 !important;
#         color: #ffffff !important;
#         border-radius: 0 !important;
#     }

#     .stTabs [aria-selected="true"] {
#         background-color: #222222 !important;
#         color: #ffffff !important;
#         border-bottom: 2px solid #ff4b4b !important;
#     }

#     /* Info / error boxes */
#     .stAlert {
#         background-color: #111111 !important;
#         color: #ffffff !important;
#     }

#     /* Dataframes */
#     .stDataFrame, .stDataFrame [class^="st"] table {
#         background-color: #000000 !important;
#         color: #ffffff !important;
#     }

#     /* Titles and text */
#     h1, h2, h3, h4, h5, h6, p, span, label {
#         color: #ffffff !important;
#     }

#     /* Buttons */
#     .stButton > button {
#         background-color: #333333 !important;
#         color: #ffffff !important;
#         border: 1px solid #666666 !important;
#     }
#     .stButton > button:hover {
#         background-color: #555555 !important;
#         border-color: #aaaaaa !important;
#     }
#     </style>
# """, unsafe_allow_html=True)


# # ========================
# # PATHS & CONFIG
# # ========================
# BASE_DIR = os.path.dirname(__file__)
# RESULTS_DIR = os.path.join(BASE_DIR, "precomputed_results")

# TASK_OPTIONS = {
#     "classical": "classicalMusic",
#     "gen1": "genMusic01",
#     "gen2": "genMusic02",
#     "gen3": "genMusic03",
# }
# TASK_LABELS = ["Classical Music", "Generative Music 1", "Generative Music 2", "Generative Music 3"]

# FREQUENCY_BANDS_SUBJECT = {
#     "Delta": {"label": "Sleep/Deep Relaxation", "cmap": "Blues"},
#     "Alpha": {"label": "Calm/Relaxation", "cmap": "Greens"},
#     "Beta": {"label": "Anxiety/Alertness", "cmap": "Reds"},
# }

# THETA_WINDOWS_ORDER = ["5s Before Onset", "At Onset", "5s After Onset"]

# # ========================
# # LOAD PRECOMPUTED DATA
# # ========================
# @st.cache_data
# def load_tab1_data():
#     path = os.path.join(RESULTS_DIR, "tab1_subject_band_powers.csv")
#     return pd.read_csv(path)

# @st.cache_data
# def load_tab3_data():
#     p1 = os.path.join(RESULTS_DIR, "tab3_gender_band_powers.csv")
#     p2 = os.path.join(RESULTS_DIR, "tab3_gender_beta_alpha_ratio.csv")
#     return pd.read_csv(p1), pd.read_csv(p2)

# @st.cache_data
# def load_tab4_data():
#     path = os.path.join(RESULTS_DIR, "tab4_theta_timecourse.csv")
#     return pd.read_csv(path)

# @st.cache_data
# def load_tab5_data():
#     p1 = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_timeseries.csv")
#     p2 = os.path.join(RESULTS_DIR, "tab5_frontal_asymmetry_summary.csv")
#     return pd.read_csv(p1), pd.read_csv(p2)

# # Utility to build fake MNE Info from channel list (for topomaps)
# def make_info_from_channels(ch_names, sfreq=250.0):
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
#     montage = mne.channels.make_standard_montage("standard_1020")
#     info.set_montage(montage, on_missing="ignore")
#     return info

# @st.cache_data
# def load_tab6_data():
#     path = os.path.join(RESULTS_DIR, "EEG_valence_arousal_by_subject.csv")
#     return pd.read_csv(path)

# # ========================
# # STREAMLIT UI
# # ========================
# st.title("🧠 EEG Music Analysis Dashboard")

# tab1, tab3, tab4, tab5, tab6 = st.tabs(
#     ["Individual Subject", "Gender Analysis", "Theta Time Course", "Frontal Asymmetry", "Arousal vs Valence"]
# )

# with tab1:
#     st.header("Individual Subject Topomaps")

#     df1 = load_tab1_data()
#     subjects = sorted(df1["subject"].unique())
#     task_keys = list(TASK_OPTIONS.keys())

#     col1, col2 = st.columns(2)
#     with col1:
#         subj = st.selectbox("Subject", subjects)
#     with col2:
#         music_key = st.selectbox("Music Type", task_keys)

#     if st.button("Show Topomaps", type="primary"):
#         sub_df = df1[(df1["subject"] == subj) & (df1["task_key"] == music_key)]
#         if sub_df.empty:
#             st.error("No data for this subject / music combination.")
#         else:
#             # fixed order
#             band_order = ["Delta", "Alpha", "Beta"]
#             ch_names = sorted(sub_df["channel"].unique())
#             info = make_info_from_channels(ch_names)

#             from matplotlib.gridspec import GridSpec

#             # single row: 3 heads + 3 colorbars + legend
#             fig = plt.figure(figsize=(18, 5))
#             gs = GridSpec(
#                 nrows=1,
#                 ncols=7,
#                 width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 0.9],
#                 figure=fig,
#             )

#             topo_axes = [
#                 fig.add_subplot(gs[0, 0]),
#                 fig.add_subplot(gs[0, 2]),
#                 fig.add_subplot(gs[0, 4]),
#             ]
#             cbar_axes = [
#                 fig.add_subplot(gs[0, 1]),
#                 fig.add_subplot(gs[0, 3]),
#                 fig.add_subplot(gs[0, 5]),
#             ]
#             legend_ax = fig.add_subplot(gs[0, 6])

#             # ----- draw the three bands with their own colorbars -----
#             for band, ax, cax in zip(band_order, topo_axes, cbar_axes):
#                 band_df = sub_df[sub_df["band"] == band]
#                 band_df = band_df.set_index("channel").reindex(ch_names)
#                 vals = band_df["power"].to_numpy()

#                 im, _ = mne.viz.plot_topomap(
#                     vals,
#                     info,
#                     axes=ax,
#                     show=False,
#                     cmap=FREQUENCY_BANDS_SUBJECT[band]["cmap"],
#                     contours=6,
#                     extrapolate="head",
#                     sphere=(0.0, 0.0, 0.09, 0.2),
#                 )

#                 # band-specific title
#                 if band == "Delta":
#                     t = "Delta Band (1–4 Hz)\nSleep/Deep Relaxation"
#                 elif band == "Alpha":
#                     t = "Alpha Band (8–13 Hz)\nCalm/Relaxation"
#                 else:
#                     t = "Beta Band (13–30 Hz)\nAnxiety/Alertness"
#                 ax.set_title(t, fontsize=12, fontweight="bold", color="white")

#                 # individual colorbar
#                 cbar = fig.colorbar(im, cax=cax)
#                 cbar.ax.tick_params(colors="white", labelsize=8)
#                 cbar.set_label("Power (µV²/Hz)", color="white", fontsize=9)

#             # ----- brain regions guide on the right -----
#             legend_ax.axis("off")
#             legend_ax.set_facecolor("#000000")
#             legend_ax.text(
#                 0.0,
#                 1.02,
#                 "Brain Regions Guide",
#                 fontsize=13,
#                 fontweight="bold",
#                 color="white",
#                 transform=legend_ax.transAxes,
#             )

#             regions = [
#                 ("Frontal (Front)", "Executive function\nDecision making\nPlanning"),
#                 ("Temporal (Sides)", "Auditory processing\nMusic perception\nMemory"),
#                 ("Parietal (Top-Back)", "Sensory integration\nSpatial awareness"),
#                 ("Occipital (Back)", "Visual processing"),
#                 ("Central", "Motor control\nSensorimotor"),
#             ]

#             y = 0.9
#             dy = 0.18
#             for title, desc in regions:
#                 legend_ax.text(
#                     0.0,
#                     y,
#                     title,
#                     fontsize=11,
#                     fontweight="bold",
#                     color="#003399",
#                     transform=legend_ax.transAxes,
#                     va="top",
#                 )
#                 legend_ax.text(
#                     0.0,
#                     y - 0.05,
#                     desc,
#                     fontsize=9,
#                     color="white",
#                     transform=legend_ax.transAxes,
#                     va="top",
#                 )
#                 y -= dy

#             # dark theme
#             fig.patch.set_facecolor("#000000")
#             for ax in topo_axes:
#                 ax.set_facecolor("#000000")
#                 ax.tick_params(colors="white")

#             plt.tight_layout()
#             st.pyplot(fig)



# # =====================================================
# # TAB 3 – Gender Analysis
# # =====================================================
# with tab3:
#     st.header("Gender-based Band Power & Beta/Alpha Ratio")
#     band_df, ratio_df = load_tab3_data()

#     if st.checkbox("Show band power plots", value=True):
#         plt.style.use('dark_background')
#         fig = plt.figure(figsize=(14, 10))
#         ax1 = plt.subplot(2, 2, 1)
#         sns.barplot(data=band_df, x="band", y="power", hue="gender", ax=ax1, errorbar="se")
#         ax1.set_title("Mean Band Power by Gender")
#         ax1.set_ylabel("Power (μV²/Hz)")

#         ax2 = plt.subplot(2, 2, 2)
#         sns.boxplot(data=band_df, x="band", y="power", hue="gender", ax=ax2)
#         ax2.set_title("Band Power Distribution")
#         ax2.set_ylabel("Power (μV²/Hz)")

#         ax3 = plt.subplot(2, 1, 2)
#         sns.barplot(data=band_df, x="task_label", y="power", hue="gender", ax=ax3, errorbar="se")
#         ax3.set_title("Band Power by Task and Gender")
#         ax3.set_ylabel("Power (μV²/Hz)")
#         ax3.tick_params(axis="x", rotation=30)
        
#         fig.patch.set_facecolor('#000000')
#         plt.tight_layout()
#         st.pyplot(fig)

#     st.markdown("---")

#     if st.checkbox("Show Beta/Alpha ratio plots", value=True):
#         plt.style.use('dark_background')
#         fig = plt.figure(figsize=(12, 6))
#         ax1 = plt.subplot(1, 2, 1)
#         sns.barplot(data=ratio_df, x="gender", y="beta_alpha_ratio", ax=ax1, errorbar="se")
#         ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
#         ax1.set_title("Beta/Alpha Ratio by Gender")
#         ax1.set_ylabel("Beta/Alpha Ratio")

#         ax2 = plt.subplot(1, 2, 2)
#         sns.barplot(data=ratio_df, x="task_label", y="beta_alpha_ratio", hue="gender", ax=ax2, errorbar="se")
#         ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
#         ax2.set_title("Beta/Alpha Ratio by Task and Gender")
#         ax2.tick_params(axis="x", rotation=30)
        
#         fig.patch.set_facecolor('#000000')
#         plt.tight_layout()
#         st.pyplot(fig)

# # =====================================================
# # TAB 4 – Theta Time Course
# # =====================================================
# with tab4:
#     st.header("Theta Band Time-Course Topomaps")
#     df4 = load_tab4_data()
#     df4["task_label"] = df4["task_bids"].map({v: lbl for v, lbl in zip(TASK_OPTIONS.values(), TASK_LABELS)})
#     tasks_choice = st.multiselect("Music Types", sorted(df4["task_label"].unique()), default=sorted(df4["task_label"].unique()))

#     if st.button("Show Theta Topomaps (Precomputed)", type="primary"):
#         sub = df4[df4["task_label"].isin(tasks_choice)]
#         if sub.empty:
#             st.error("No data for selected tasks.")
#         else:
#             ch_names = sorted(sub["channel"].unique())
#             info = make_info_from_channels(ch_names)
#             n_tasks = len(tasks_choice)
#             n_times = len(THETA_WINDOWS_ORDER)
#             fig, axes = plt.subplots(n_tasks, n_times, figsize=(5 * n_times, 4 * n_tasks))
#             if n_tasks == 1:
#                 axes = np.array([axes])

#             all_vals = []
#             theta_maps = {t: {} for t in tasks_choice}
#             for t in tasks_choice:
#                 tdf = sub[sub["task_label"] == t]
#                 for w in THETA_WINDOWS_ORDER:
#                     wdf = tdf[tdf["window"] == w]
#                     if wdf.empty:
#                         continue
#                     wdf = wdf.set_index("channel").reindex(ch_names)
#                     vals = wdf["theta_power"].to_numpy()
#                     theta_maps[t][w] = vals
#                     all_vals.append(vals)

#             if not all_vals:
#                 st.error("Not enough data for selected tasks/windows.")
#             else:
#                 flat = np.concatenate(all_vals)
#                 vmin, vmax = np.nanpercentile(flat, [5, 95])
#                 last_im = None
#                 for ti, t in enumerate(tasks_choice):
#                     for wi, w in enumerate(THETA_WINDOWS_ORDER):
#                         ax = axes[ti, wi]
#                         if w not in theta_maps[t]:
#                             ax.axis("off")
#                             continue
#                         vals = theta_maps[t][w]
#                         im, _ = mne.viz.plot_topomap(
#                             vals, info, axes=ax, show=False, vlim=(vmin, vmax),
#                             cmap="viridis", contours=6, extrapolate="head", sphere="auto",
#                         )
#                         last_im = im
#                         if ti == 0:
#                             ax.set_title(w, fontsize=11, fontweight="bold", color='white')
#                         if wi == 0:
#                             ax.set_ylabel(t, fontsize=11, fontweight="bold", color='white')

#                 if last_im is not None:
#                     fig.subplots_adjust(right=0.90)
#                     cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
#                     cbar = fig.colorbar(last_im, cax=cax)
#                     cbar.set_label("Theta Power (μV²/Hz)", fontsize=11, color='white')
#                     cbar.ax.tick_params(colors='white')
                
#                 # Dark theme
#                 fig.patch.set_facecolor('#000000')
#                 st.pyplot(fig)

# # =====================================================
# # TAB 5 – Frontal Alpha Asymmetry
# # =====================================================
# with tab5:
#     st.header("Frontal Alpha Asymmetry (Classical Music)")
#     ts_df, summary_df = load_tab5_data()
#     pairs = sorted(ts_df["pair"].unique())
#     pair = st.selectbox("Electrode pair", pairs)

#     pair_df = ts_df[ts_df["pair"] == pair]
#     if pair_df.empty:
#         st.error("No data for selected pair.")
#     else:
#         subjects = sorted(pair_df["subject"].unique())
#         times = np.sort(pair_df["time"].unique())
#         mat = np.zeros((len(subjects), len(times)))
#         sub_to_row = {s: i for i, s in enumerate(subjects)}
#         time_to_col = {t: j for j, t in enumerate(times)}

#         for _, row in pair_df.iterrows():
#             i = sub_to_row[row["subject"]]
#             j = time_to_col[row["time"]]
#             mat[i, j] = row["asymmetry"]

#         plt.style.use('dark_background')
#         fig = plt.figure(figsize=(14, 6))
#         ax1 = plt.subplot(1, 2, 1)
#         vmax = np.percentile(np.abs(mat), 95)
#         vmin = -vmax
#         im = ax1.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
#                        extent=[times[0], times[-1], len(subjects) - 0.5, -0.5])
#         ax1.set_xlabel("Time (s)", color='white')
#         ax1.set_ylabel("Subject index", color='white')
#         ax1.set_title(f"{pair} Asymmetry (Right - Left)", color='white')
#         cbar = plt.colorbar(im, ax=ax1)
#         cbar.set_label("Asymmetry", color='white')
#         cbar.ax.tick_params(colors='white')

#         ax2 = plt.subplot(1, 2, 2)
#         mean_asym = np.mean(mat, axis=0)
#         std_asym = np.std(mat, axis=0)
#         sem = std_asym / np.sqrt(mat.shape[0])
#         ax2.plot(times, mean_asym, "cyan", label="Mean asymmetry", linewidth=2)
#         ax2.fill_between(times, mean_asym - 1.96 * sem, mean_asym + 1.96 * sem,
#                         alpha=0.3, color="cyan", label="95% CI")
#         ax2.axhline(0, color="white", linestyle="--", linewidth=1)
#         ax2.set_xlabel("Time (s)", color='white')
#         ax2.set_ylabel("Asymmetry (R-L)", color='white')
#         ax2.set_title(f"{pair} Mean Asymmetry Over Time", color='white')
#         ax2.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
#         ax2.tick_params(colors='white')
        
#         fig.patch.set_facecolor('#000000')
#         plt.tight_layout()
#         st.pyplot(fig)

#         st.markdown("### Summary statistics")
#         st.dataframe(summary_df[summary_df["pair"] == pair])

# # =====================================================
# # TAB 6 – Arousal vs Valence
# # =====================================================
# with tab6:
#     st.header("Arousal vs Valence (Per Subject & Task)")
#     df6 = load_tab6_data()
#     df_plot = df6.rename(columns={"subject": "Subject", "task_bids": "Task", "valence": "Valence", "arousal": "Arousal"})

#     if df_plot.empty:
#         st.error("No precomputed arousal/valence data found.")
#     else:
#         tasks = df_plot["Task"].unique()
#         subjects = df_plot["Subject"].unique()
#         task_name_map = {"classicalMusic": "Classical Music", "genMusic01": "Generative Music 1", 
#                         "genMusic02": "Generative Music 2", "genMusic03": "Generative Music 3"}
#         palette = sns.color_palette("hsv", n_colors=len(subjects))
#         markers = ["o", "s", "D", "^", "v"]
#         marker_map = {sub: markers[i % len(markers)] for i, sub in enumerate(subjects)}
#         VALENCE_LIM = (-6e-10, 6e-10)
#         AROUSAL_LIM = (-10, 130)

#         explanation_note = (
#             "This chart maps emotional responses to music using a 2D arousal–valence model.\n\n"
#             "• **Valence (X-axis)**: Emotional direction / pleasantness (right = more positive/joyful, left = more negative/sad).\n\n"
#             "• **Arousal (Y-axis)**: Emotional intensity (up = higher intensity/tension, down = calmer/relaxed)."
#         )
#         st.info(explanation_note)

#         plt.style.use('dark_background')
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         axes = axes.flatten()
#         fig.suptitle("Arousal vs Valence by Music Type", fontsize=18, y=0.95, color='white')

#         legend_handles = []
#         legend_labels = []

#         for i, task in enumerate(tasks):
#             if i >= len(axes):
#                 break
#             ax = axes[i]
#             subset = df_plot[df_plot["Task"] == task]
#             display_name = task_name_map.get(task, task)

#             for j, subject in enumerate(subjects):
#                 sub_data = subset[subset["Subject"] == subject]
#                 if sub_data.empty:
#                     continue
#                 valence_mean = sub_data["Valence"].mean()
#                 arousal_mean = sub_data["Arousal"].mean()
#                 handle = ax.scatter(valence_mean, arousal_mean, color=palette[j], s=80, alpha=0.8,
#                                   marker=marker_map[subject], label=subject, edgecolors='white', linewidth=0.5)
#                 if i == 0:
#                     legend_handles.append(handle)
#                     legend_labels.append(subject)

#             if i % 2 == 0:
#                 ax.set_ylabel("Arousal (Intensity / Emotional Strength)", fontsize=12, color='white')
#             if i >= 2:
#                 ax.set_xlabel("Valence (Pleasantness / Emotional Direction)", fontsize=12, color='white')
#             ax.set_title(display_name, fontsize=14, color='white')
#             ax.set_xlim(VALENCE_LIM)
#             ax.set_ylim(AROUSAL_LIM)
#             ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
#             ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
#             ax.grid(True, linestyle=":", alpha=0.6)
#             ax.tick_params(colors='white')

#         if len(tasks) < 4:
#             for k in range(len(tasks), 4):
#                 axes[k].axis("off")

#         plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
#         # Dark theme styling
#         fig.patch.set_facecolor('#000000')
#         for ax in axes.flat:
#             ax.set_facecolor('#111111')
        
#         if legend_handles:
#             fig.legend(legend_handles, legend_labels, loc='center right', bbox_to_anchor=(0.99, 0.5),
#                       fontsize=10, title="Subject ID", title_fontsize=12,
#                       facecolor='#222222', edgecolor='#444444', labelcolor='white')

#         st.pyplot(fig)

#         with st.expander("Show raw arousal/valence data"):
#             st.dataframe(df6)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import streamlit as st


st.set_page_config(page_title="EEG Music Analysis Dashboard", layout="wide")


st.markdown("""
    <style>
    /* Make whole page black */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Main app block */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Top header area */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }

    /* Tabs bar */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #111111 !important;
        color: #ffffff !important;
        border-radius: 0 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #222222 !important;
        color: #ffffff !important;
        border-bottom: 2px solid #ff4b4b !important;
    }

    /* Info / error boxes */
    .stAlert {
        background-color: #111111 !important;
        color: #ffffff !important;
    }

    /* Dataframes */
    .stDataFrame, .stDataFrame [class^="st"] table {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Titles and text */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #666666 !important;
    }
    .stButton > button:hover {
        background-color: #555555 !important;
        border-color: #aaaaaa !important;
    }
    </style>
""", unsafe_allow_html=True)



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


# ========================
# LOAD PRECOMPUTED DATA
# ========================
@st.cache_data
def load_tab1_data():
    path = os.path.join(RESULTS_DIR, "tab1_subject_band_powers.csv")
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


@st.cache_data
def load_tab6_data():
    path = os.path.join(RESULTS_DIR, "EEG_valence_arousal_by_subject.csv")
    return pd.read_csv(path)


# ========================
# STREAMLIT UI
# ========================
st.title("🧠 EEG Music Analysis Dashboard")


tab1, tab3, tab4, tab5, tab6 = st.tabs(
    ["Individual Participant", "Gender Analysis", "Theta Time Course", "Frontal Asymmetry", "Arousal vs Valence"]
)


with tab1:
    st.header("Individual Participant Topomaps")

    df1 = load_tab1_data()
    participants = sorted(df1["subject"].unique())
    task_keys = list(TASK_OPTIONS.keys())

    col1, col2 = st.columns(2)
    with col1:
        subj = st.selectbox("Participant", participants
