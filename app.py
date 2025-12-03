#####TRial 3 ####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import streamlit as st
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header

# ========================
# PAGE CONFIG & THEME
# ========================
st.set_page_config(page_title="EEG Music Analysis Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #05070B !important;
        color: #F5F5F7 !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }

    /* Main app block */
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 1rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        background: radial-gradient(circle at top left, #101320 0, #05070B 40%, #000000 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #05070B 0%, #080B15 40%, #05070B 100%) !important;
        color: #F5F5F7 !important;
    }

    /* Top header area */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* Tabs bar */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(10, 12, 20, 0.9) !important;
        border-bottom: 1px solid #22232F;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #AAAAAE !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500;
        border-radius: 0 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #222433, #181A25) !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #FF4B4B !important;
    }

    /* Info / error boxes */
    .stAlert {
        background: radial-gradient(circle at top left, #141729 0, #0B0D18 45%, #060712 100%) !important;
        color: #F5F5F7 !important;
        border-radius: 0.75rem !important;
        border: 1px solid #262A3C !important;
    }

    /* Dataframes */
    .stDataFrame, .stDataFrame [class^="st"] table {
        background-color: #05070B !important;
        color: #F5F5F7 !important;
    }

    /* Titles and text */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        letter-spacing: 0.02em;
    }
    p, span, label {
        color: #D6D7DD !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2D2F3D, #393C4E) !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        border: 1px solid #5C5F7A !important;
        padding: 0.4rem 1.4rem !important;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #FF4B4B, #FF7A3C) !important;
        border-color: #FF9B6A !important;
    }

    /* Cards under headers */
    .section-card {
        background: radial-gradient(circle at top left, #171A2A 0, #0C0F1C 45%, #05070B 100%);
        border-radius: 1rem;
        border: 1px solid #272A3B;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #A9ABB6;
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

@st.cache_data
def load_tab6_data():
    path = os.path.join(RESULTS_DIR, "EEG_valence_arousal_by_subject.csv")
    return pd.read_csv(path)

# Utility to build fake MNE Info from channel list (for topomaps)
def make_info_from_channels(ch_names, sfreq=250.0):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info

# ========================
# HEADER & TABS
# ========================
title_col1, title_col2 = st.columns([3, 1])
with title_col1:
    st.markdown(
        "<h1 style='margin-bottom:0.1rem;'>🧠 EEG Music Analysis Dashboard</h1>"
        "<p class='section-subtitle'>Explore how different musical styles shape neural activity, "
        "cognitive states, and emotional responses across participants.</p>",
        unsafe_allow_html=True,
    )
with title_col2:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Participants", "21")
    with c2:
        st.metric("Music Conditions", "4")
style_metric_cards(
    background_color="#111320",
    border_color="#313852",
    border_left_color="#FF4B4B",
)

tab1, tab3, tab4, tab5, tab6 = st.tabs(
    ["Individual Participant", "Gender Analysis", "Theta Time Course", "Frontal Asymmetry", "Arousal vs Valence"]
)

# =====================================================
# TAB 1 – Individual Participant Topomaps
# =====================================================
with tab1:
    colored_header(
        label="Individual Participant Topomaps",
        description="Visualize spatial EEG band power patterns for a selected participant and music condition.",
        color_name="red-70",
    )

    df1 = load_tab1_data()

    available_pairs = (
        df1[["subject", "task_key"]]
        .drop_duplicates()
        .sort_values(["subject", "task_key"])
    )

    if available_pairs.empty:
        st.error("No participant data available.")
        st.stop()

    participants = sorted(available_pairs["subject"].unique())

    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        subj = st.selectbox(
            "Participant",
            participants,
            help="Choose a participant to inspect their EEG topomaps.",
        )
    valid_tasks_for_subj = available_pairs.loc[
        available_pairs["subject"] == subj, "task_key"
    ].tolist()
    with ctrl_col2:
        music_key = st.selectbox(
            "Music Type",
            valid_tasks_for_subj,
            help="Select the music condition for this participant.",
        )

    if st.button("Show Topomaps", type="primary"):
        sub_df = df1[(df1["subject"] == subj) & (df1["task_key"] == music_key)]

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
                t = "Delta Band (1–4 Hz)\nSleep / Deep Relaxation"
            elif band == "Alpha":
                t = "Alpha Band (8–13 Hz)\nCalm / Relaxation"
            else:
                t = "Beta Band (13–30 Hz)\nAnxiety / Alertness"
            ax.set_title(t, fontsize=12, fontweight="bold", color="white")

            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(colors="white", labelsize=8)
            cbar.set_label("Power (µV²/Hz)", color="white", fontsize=9)

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
                color="#45a2ff",
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

# =====================================================
# TAB 3 – Gender Analysis (interactive)
# =====================================================
with tab3:
    colored_header(
        label="Gender-based Band Power & Ratios",
        description="Compare spectral power and beta/alpha balance across genders and music types.",
        color_name="violet-70",
    )

    band_df, ratio_df = load_tab3_data()

    sub_tab1, sub_tab2 = st.tabs(["Band Power", "Beta/Alpha Ratio"])

    with sub_tab1:
        fig1 = px.bar(
            band_df,
            x="band",
            y="power",
            color="gender",
            barmode="group",
            hover_data={"power": ":3e", "band": True, "gender": True, "task_label": True},
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
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )

        fig2 = px.bar(
            band_df,
            x="task_label",
            y="power",
            color="gender",
            barmode="group",
            hover_data={"power": ":3e", "band": True},
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
            xaxis_tickangle=-30,
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(fig1, use_container_width=True)
        with col_b:
            st.plotly_chart(fig2, use_container_width=True)

    with sub_tab2:
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
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
            showlegend=False,
        )

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
            xaxis_tickangle=-30,
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
            legend=dict(
                bgcolor="#111111",
                bordercolor="#444444",
                borderwidth=1,
                font=dict(color="white"),
                title_font=dict(color="white"),
            ),
        )

        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(fig3, use_container_width=True)
        with col_d:
            st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# TAB 4 – Theta Time Course
# =====================================================
with tab4:
    colored_header(
        label="Theta Band Time-Course Topomaps",
        description="Inspect how theta power evolves around music onset across tasks.",
        color_name="blue-70",
    )

    df4 = load_tab4_data()
    df4["task_label"] = df4["task_bids"].map({v: lbl for v, lbl in zip(TASK_OPTIONS.values(), TASK_LABELS)})
    tasks_choice = st.multiselect(
        "Music Types",
        sorted(df4["task_label"].unique()),
        default=sorted(df4["task_label"].unique()),
    )

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
                            ax.set_title(w, fontsize=11, fontweight="bold", color="white")
                        if wi == 0:
                            ax.set_ylabel(t, fontsize=11, fontweight="bold", color="white")

                if last_im is not None:
                    fig.subplots_adjust(right=0.90)
                    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
                    cbar = fig.colorbar(last_im, cax=cax)
                    cbar.set_label("Theta Power (μV²/Hz)", fontsize=11, color="white")
                    cbar.ax.tick_params(colors="white")

                fig.patch.set_facecolor("#000000")
                st.pyplot(fig)

# =====================================================
# TAB 5 – Frontal Alpha Asymmetry
# =====================================================
with tab5:
    colored_header(
        label="Frontal Alpha Asymmetry (Classical Music)",
        description="Examine left–right frontal activation differences over time.",
        color_name="green-70",
    )

    ts_df, summary_df = load_tab5_data()
    pairs = sorted(ts_df["pair"].unique())
    pair = st.selectbox("Electrode pair", pairs)

    pair_df = ts_df[ts_df["pair"] == pair]
    if pair_df.empty:
        st.error("No data for selected pair.")
    else:
        participants = sorted(pair_df["subject"].unique())
        times = np.sort(pair_df["time"].unique())

        mat = np.zeros((len(participants), len(times)))
        sub_to_row = {s: i for i, s in enumerate(participants)}
        time_to_col = {t: j for j, t in enumerate(times)}

        for _, row in pair_df.iterrows():
            i = sub_to_row[row["subject"]]
            j = time_to_col[row["time"]]
            mat[i, j] = row["asymmetry"]

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
                title=dict(text="Asymmetry", font=dict(color="white")),
                tickcolor="white",
                tickfont=dict(color="white"),
            ),
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        mean_asym = np.mean(mat, axis=0)
        std_asym = np.std(mat, axis=0)
        sem = std_asym / np.sqrt(mat.shape[0])

        mean_df = pd.DataFrame({"time": times, "mean_asym": mean_asym, "sem": sem})

        fig_line = px.line(
            mean_df,
            x="time",
            y="mean_asym",
            labels={"time": "Time (s)", "mean_asym": "Asymmetry (R-L)"},
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
            title=dict(font=dict(color="white", size=18), x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### Summary statistics")
        st.dataframe(
            summary_df[summary_df["pair"] == pair],
            use_container_width=True,
            hide_index=True,
        )

# =====================================================
# TAB 6 – Arousal vs Valence (interactive)
# =====================================================
with tab6:
    colored_header(
        label="Arousal vs Valence",
        description="Map average emotional responses to different music types for each participant.",
        color_name="orange-70",
    )

    df6 = load_tab6_data()

    if df6.empty:
        st.error("No arousal/valence data found.")
    else:
        df_grp = (
            df6.groupby(["Subject", "Task"], as_index=False)[["Valence", "Arousal"]]
            .mean()
        )

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

        participants = sorted(df_plot["ParticipantID"].unique())
        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            participants_selected = st.multiselect(
                "Participants",
                participants,
                default=participants,
            )
        with ctrl2:
            st.write("")
            st.write(f"**Selected:** {len(participants_selected)} / {len(participants)}")

        sub = df_plot[df_plot["ParticipantID"].isin(participants_selected)]

        if sub.empty:
            st.error("No data for the selected participants.")
        else:
            fig = px.scatter(
                sub,
                x="Valence",
                y="Arousal",
                color="ParticipantID",
                facet_col="Task Label",
                facet_col_wrap=2,
                height=800,  # <-- add this (try 700–900)
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


            fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(color="white", width=0.4)))

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

            fig.update_layout(
                grid=dict(
                    rows=2,
                    columns=2,
                    pattern="independent",
                    roworder="top to bottom",
                    xgap=0.10,
                    ygap=0.105,
                )
            )

            fig.for_each_annotation(
                lambda a: a.update(font=dict(color="#E5E5EA", size=13))
            )

            st.markdown(
                """
                The scatter plots visualize the average emotional impact of different types of music as observed in 21 participants.  
                From the visualization, classical music shows the most diverse range of emotional responses, while generative music
                responses are more tightly clustered, indicating comparatively lower emotional variability.
                
                
                ⁠Valence (X-axis): Measures the Emotional Direction (Pleasantness).
                Right = Positive/Joyful; Left = Negative/Sadness.
                
                
                ⁠Arousal (Y-axis): Measures the Emotional Intensity (Strength/Energy).
                Up = High Intensity/Tension; Down = Low Intensity/Calmness.
                """
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show aggregated table (per participant & task)"):
            st.dataframe(df_plot, use_container_width=True, hide_index=True)
