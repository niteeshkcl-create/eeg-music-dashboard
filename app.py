

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import streamlit as st
import plotly.express as px


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

    # Build only valid (participant, task_key) combinations
    available_pairs = (
        df1[["subject", "task_key"]]
        .drop_duplicates()
        .sort_values(["subject", "task_key"])
    )

    # If nothing at all, show message and stop
    if available_pairs.empty:
        st.error("No participant data available.")
        st.stop()

    participants = sorted(available_pairs["subject"].unique())

    col1, col2 = st.columns(2)
    with col1:
        subj = st.selectbox("Participant", participants)

    # Filter tasks for the chosen participant only
    valid_tasks_for_subj = available_pairs.loc[
        available_pairs["subject"] == subj, "task_key"
    ].tolist()

    with col2:
        music_key = st.selectbox("Music Type", valid_tasks_for_subj)

    if st.button("Show Topomaps", type="primary"):
        # Now this combo is guaranteed to exist
        sub_df = df1[(df1["subject"] == subj) & (df1["task_key"] == music_key)]

        # fixed order
        band_order = ["Delta", "Alpha", "Beta"]
        ch_names = sorted(sub_df["channel"].unique())
        info = make_info_from_channels(ch_names)

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(
            nrows=1,
            ncols=7,
            width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 0.9],
            figure=fig,
        )

        topo_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 4]),
        ]
        cbar_axes = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[0, 5]),
        ]
        legend_ax = fig.add_subplot(gs[0, 6])

        # ----- draw the three bands with their own colorbars -----
        for band, ax, cax in zip(band_order, topo_axes, cbar_axes):
            band_df = sub_df[sub_df["band"] == band]
            band_df = band_df.set_index("channel").reindex(ch_names)
            vals = band_df["power"].to_numpy()

            im, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=ax,
                show=False,
                cmap=FREQUENCY_BANDS_SUBJECT[band]["cmap"],
                contours=6,
                extrapolate="head",
                sphere=(0.0, 0.0, 0.09, 0.2),
            )

            if band == "Delta":
                t = "Delta Band (1–4 Hz)\nSleep/Deep Relaxation"
            elif band == "Alpha":
                t = "Alpha Band (8–13 Hz)\nCalm/Relaxation"
            else:
                t = "Beta Band (13–30 Hz)\nAnxiety/Alertness"
            ax.set_title(t, fontsize=12, fontweight="bold", color="white")

            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(colors="white", labelsize=8)
            cbar.set_label("Power (µV²/Hz)", color="white", fontsize=9)

        # ----- brain regions guide on the right -----
        legend_ax.axis("off")
        legend_ax.set_facecolor("#000000")
        legend_ax.text(
            0.0,
            1.02,
            "Brain Regions Guide",
            fontsize=13,
            fontweight="bold",
            color="white",
            transform=legend_ax.transAxes,
        )

        regions = [
            ("Frontal (Front)", "Executive function\nDecision making\nPlanning"),
            ("Temporal (Sides)", "Auditory processing\nMusic perception\nMemory"),
            ("Parietal (Top-Back)", "Sensory integration\nSpatial awareness"),
            ("Occipital (Back)", "Visual processing"),
            ("Central", "Motor control\nSensorimotor"),
        ]

        y = 0.9
        dy = 0.18
        for title, desc in regions:
            legend_ax.text(
                0.0,
                y,
                title,
                fontsize=11,
                fontweight="bold",
                color="#003399",
                transform=legend_ax.transAxes,
                va="top",
            )
            legend_ax.text(
                0.0,
                y - 0.05,
                desc,
                fontsize=9,
                color="white",
                transform=legend_ax.transAxes,
                va="top",
            )
            y -= dy

        fig.patch.set_facecolor("#000000")
        for ax in topo_axes:
            ax.set_facecolor("#000000")
            ax.tick_params(colors="white")

        plt.tight_layout()
        st.pyplot(fig)


with tab3:
    st.header("Gender-based Band Power & Beta/Alpha Ratio")
    band_df, ratio_df = load_tab3_data()

    # ---------- Interactive band power plots ----------
    if st.checkbox("Show band power plots", value=True):
        # Mean band power by gender
        fig1 = px.bar(
            band_df,
            x="band",
            y="power",
            color="gender",
            barmode="group",
            hover_data={"power": ":.3e", "band": True, "gender": True, "task_label": True},
            labels={
                "band": "Band",
                "power": "Power (μV²/Hz)",
                "gender": "Gender",
            },
            title="Mean Band Power by Gender",
        )
        fig1.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Band power by task and gender
        fig2 = px.bar(
            band_df,
            x="task_label",
            y="power",
            color="gender",
            barmode="group",
            hover_data={"power": ":.3e", "band": True},
            labels={
                "task_label": "Music Type",
                "power": "Power (μV²/Hz)",
                "gender": "Gender",
            },
            title="Band Power by Task and Gender",
        )
        fig2.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ---------- Interactive beta/alpha ratio plots ----------
    if st.checkbox("Show Beta/Alpha ratio plots", value=True):
        # Ratio by gender
        fig3 = px.bar(
            ratio_df,
            x="gender",
            y="beta_alpha_ratio",
            hover_data={"beta_alpha_ratio": ":.3f"},
            labels={
                "gender": "Gender",
                "beta_alpha_ratio": "Beta/Alpha Ratio",
            },
            title="Beta/Alpha Ratio by Gender",
        )
        fig3.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig3.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Ratio by task and gender
        fig4 = px.bar(
            ratio_df,
            x="task_label",
            y="beta_alpha_ratio",
            color="gender",
            barmode="group",
            hover_data={"beta_alpha_ratio": ":.3f"},
            labels={
                "task_label": "Music Type",
                "beta_alpha_ratio": "Beta/Alpha Ratio",
                "gender": "Gender",
            },
            title="Beta/Alpha Ratio by Task and Gender",
        )
        fig4.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig4.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )
        st.plotly_chart(fig4, use_container_width=True)



# =====================================================
# TAB 4 – Theta Time Course
# =====================================================
with tab4:
    st.header("Theta Band Time-Course Topomaps")
    df4 = load_tab4_data()
    df4["task_label"] = df4["task_bids"].map({v: lbl for v, lbl in zip(TASK_OPTIONS.values(), TASK_LABELS)})
    tasks_choice = st.multiselect("Music Types", sorted(df4["task_label"].unique()), default=sorted(df4["task_label"].unique()))

    if st.button("Show Theta Topomaps", type="primary"):
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
                            vals, info, axes=ax, show=False, vlim=(vmin, vmax),
                            cmap="viridis", contours=6, extrapolate="head", sphere="auto",
                        )
                        last_im = im
                        if ti == 0:
                            ax.set_title(w, fontsize=11, fontweight="bold", color='white')
                        if wi == 0:
                            ax.set_ylabel(t, fontsize=11, fontweight="bold", color='white')

                if last_im is not None:
                    fig.subplots_adjust(right=0.90)
                    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
                    cbar = fig.colorbar(last_im, cax=cax)
                    cbar.set_label("Theta Power (μV²/Hz)", fontsize=11, color='white')
                    cbar.ax.tick_params(colors='white')
                
                fig.patch.set_facecolor('#000000')
                st.pyplot(fig)


# =====================================================
# TAB 5 – Frontal Alpha Asymmetry
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
        # pivot to subjects x time matrix
        participants = sorted(pair_df["subject"].unique())
        times = np.sort(pair_df["time"].unique())

        mat = np.zeros((len(participants), len(times)))
        sub_to_row = {s: i for i, s in enumerate(participants)}
        time_to_col = {t: j for j, t in enumerate(times)}

        for _, row in pair_df.iterrows():
            i = sub_to_row[row["subject"]]
            j = time_to_col[row["time"]]
            mat[i, j] = row["asymmetry"]

        # ---------- Interactive heatmap ----------
        heat_df = pd.DataFrame(mat, index=participants, columns=times)

        vmax = np.percentile(np.abs(mat), 95)
        vmin = -vmax

        fig_heat = px.imshow(
            heat_df,
            x=times,
            y=participants,
            zmin=vmin,
            zmax=vmax,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels={
                "x": "Time (s)",
                "y": "Participant",
                "color": "Asymmetry",
            },
            title=f"{pair} Asymmetry (Right - Left)",
        )
        fig_heat.update_layout(
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            coloraxis_colorbar=dict(
                title=dict(
                    text="Asymmetry",
                    font=dict(color="white"),
                ),
                tickcolor="white",
                tickfont=dict(color="white"),
            ),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---------- Interactive mean-asymmetry line ----------
        mean_asym = np.mean(mat, axis=0)
        std_asym = np.std(mat, axis=0)
        sem = std_asym / np.sqrt(mat.shape[0])

        mean_df = pd.DataFrame({"time": times, "mean_asym": mean_asym, "sem": sem})

        fig_line = px.line(
            mean_df,
            x="time",
            y="mean_asym",
            labels={
                "time": "Time (s)",
                "mean_asym": "Asymmetry (R-L)",
            },
            title=f"{pair} Mean Asymmetry Over Time (Interactive)",
        )
        fig_line.update_traces(
            mode="lines+markers",
            line=dict(color="cyan"),
            hovertemplate="t=%{x:.2f}s<br>asym=%{y:.4f}<extra></extra>",
        )
        fig_line.add_scatter(
            x=mean_df["time"],
            y=mean_df["mean_asym"] + 1.96 * mean_df["sem"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
        fig_line.add_scatter(
            x=mean_df["time"],
            y=mean_df["mean_asym"] - 1.96 * mean_df["sem"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,255,255,0.2)",
            showlegend=False,
            hoverinfo="skip",
        )
        fig_line.add_hline(y=0.0, line_dash="dash", line_color="gray")

        fig_line.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#000000",
            font=dict(color="white"),
            title=dict(
                font=dict(color="white", size=18),
                x=0.5,
                xanchor="center",
            ),
        )

        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### Summary statistics")
        st.dataframe(summary_df[summary_df["pair"] == pair])

with tab6:
    st.header("Arousal vs Valence (Per Participant & Task)")
    df6 = load_tab6_data()  # Subject, Task, Trial, Valence, Arousal

    if df6.empty:
        st.error("No arousal/valence data found.")
    else:
        # 1) aggregate: mean per (Subject, Task)
        df_grp = (
            df6.groupby(["Subject", "Task"], as_index=False)[["Valence", "Arousal"]]
            .mean()
        )

        # 2) normalize column names
        df_plot = df_grp.rename(
            columns={
                "Subject": "ParticipantID",
                "Task": "Task",
            }
        )

        task_name_map = {
            "classicalMusic": "Classical Music",
            "genMusic01": "Generative Music 1",
            "genMusic02": "Generative Music 2",
            "genMusic03": "Generative Music 3",
        }
        df_plot["Task Label"] = df_plot["Task"].map(task_name_map).fillna(df_plot["Task"])

        # optional participant filter
        participants = sorted(df_plot["ParticipantID"].unique())
        participants_selected = st.multiselect(
            "Participants",
            participants,
            default=participants,
        )
        sub = df_plot[df_plot["ParticipantID"].isin(participants_selected)]

        if sub.empty:
            st.error("No data for the selected participants.")
        else:
            fig = px.scatter(
                sub,
                x="Valence",
                y="Arousal",
                color="ParticipantID",
                facet_col="Task Label",   # 4 music types
                facet_col_wrap=2,         # 2 graphs per row
                hover_data={
                    "ParticipantID": True,
                    "Task Label": True,
                    "Valence": ":.3e",
                    "Arousal": ":.2f",
                },
                labels={
                    "Valence": "Valence (Pleasantness)",
                    "Arousal": "Arousal (Intensity / Emotional Strength)",
                    "ParticipantID": "Participant",
                    "Task Label": "Music Type",
                },
                title="Arousal vs Valence by Participant and Music Type",
            )

            fig.update_xaxes(range=[-6e-10, 6e-10], zeroline=True, zerolinecolor="gray")
            fig.update_yaxes(range=[-10, 130], zeroline=True, zerolinecolor="gray")

            fig.update_layout(
                plot_bgcolor="#111111",
                paper_bgcolor="#000000",
                font=dict(color="white"),
                title=dict(
                    text="Arousal vs Valence by Participant and Music Type",
                    font=dict(color="white", size=18),
                    x=0.5,
                    xanchor="center",
                ),
                legend=dict(
                    title="Participant",
                    bgcolor="#111111",
                    bordercolor="#444444",
                    borderwidth=1,
                    font=dict(color="white"),
                    title_font=dict(color="white"),
                ),
                margin=dict(t=80, l=60, r=80, b=60),
            )

            # set spacing between the 2x2 facet panels
            fig.update_layout(
                grid=dict(
                    rows=2,
                    columns=2,
                    pattern="independent",
                    roworder="top to bottom",
                    xgap=0.10,   # horizontal spacing between columns
                    ygap=0.15,   # vertical spacing between rows
                )
            )


            st.markdown(
                """
                The scatter plots visualize the average emotional impact of different types of music as observed in 21 participants.  
                From the visualization, classical music shows the most diverse range of emotional responses, while generative music
                responses are more tightly clustered, indicating comparatively lower emotional variability.
                """
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show aggregated table (per participant & task)"):
            st.dataframe(df_plot)
