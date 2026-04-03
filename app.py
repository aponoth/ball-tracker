import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import time
import pandas as pd
import os
import plotly.graph_objects as go

# Constants
MIN_BALL_AREA = 100  # minimum contour area in pixels
MIN_TRAJECTORY_LENGTH = 5  # minimum points for valid trajectory
MAX_FLIGHT_TIME_SEC = 3  # maximum seconds to track a ball
GRAPH_UPDATE_INTERVAL = 10  # frames between chart updates
MIN_VELOCITY = 10  # px/s - minimum valid velocity (default, overridden by slider)
SPATIAL_FILTER_MULTIPLIER = 3.0
MIN_LAUNCH_ZONE_RADIUS = 50  # pixels
MAX_LAUNCH_ZONE_RADIUS = 200  # pixels
GRAVITY_ACCEL = 0.5  # pixels per frame squared (estimated)

# --- Initialize session state for persistent data (Must be before UI controls) ---
if 'raw_trajectories' not in st.session_state:
    st.session_state.raw_trajectories = []
if 'raw_ball_log' not in st.session_state:
    st.session_state.raw_ball_log = []
if 'all_trajectories' not in st.session_state:
    st.session_state.all_trajectories = []
if 'ball_log' not in st.session_state:
    st.session_state.ball_log = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = "video"
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'pause_frame' not in st.session_state:
    st.session_state.pause_frame = 0
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'active_tracks' not in st.session_state:
    st.session_state.active_tracks = []
if 'next_ball_id' not in st.session_state:
    st.session_state.next_ball_id = 1
if 'launch_zone_center' not in st.session_state:
    st.session_state.launch_zone_center = None
if 'launch_zone_radius' not in st.session_state:
    st.session_state.launch_zone_radius = None
if 'detection_snapshots' not in st.session_state:
    st.session_state.detection_snapshots = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'files_saved' not in st.session_state:
    st.session_state.files_saved = False
if 'saved_csv' not in st.session_state:
    st.session_state.saved_csv = None
if 'saved_chart' not in st.session_state:
    st.session_state.saved_chart = None
if 'saved_pdf' not in st.session_state:
    st.session_state.saved_pdf = None
if 'target_height_pct' not in st.session_state:
    st.session_state.target_height_pct = 40.0
if 'chart_y_min' not in st.session_state:
    st.session_state.chart_y_min = 0
if 'chart_y_max' not in st.session_state:
    st.session_state.chart_y_max = 1080
if 'video_view_type' not in st.session_state:
    st.session_state.video_view_type = "Side View"
if 'video_width' not in st.session_state:
    st.session_state.video_width = 1920
if 'video_height' not in st.session_state:
    st.session_state.video_height = 1080
if 'video_fps' not in st.session_state:
    st.session_state.video_fps = 30

st.set_page_config(layout="wide")
st.title("🏹 Advanced Ball Analytics Dashboard")

# --- Helper Functions (App State) ---

def reset_app_state(rerun=True):
    """Resets all session state variables to their initial values."""
    keys_to_reset = [
        'raw_trajectories', 'raw_ball_log', 'all_trajectories', 'ball_log',
        'active_tracks', 'pause_frame', 'paused', 'next_ball_id',
        'last_frame', 'launch_zone_center', 'launch_zone_radius',
        'analysis_complete', 'files_saved', 'saved_csv', 'saved_chart', 'saved_pdf',
        'detection_snapshots', 'video_view_type', 'target_height_pct',
        'chart_y_min', 'chart_y_max'
    ]
    for key in keys_to_reset:
        if key in ['raw_trajectories', 'raw_ball_log', 'all_trajectories', 'ball_log', 'active_tracks', 'detection_snapshots']:
            st.session_state[key] = []
        elif key in ['pause_frame', 'next_ball_id']:
            st.session_state[key] = 0 if key == 'pause_frame' else 1
        elif key in ['paused', 'analysis_complete', 'files_saved']:
            st.session_state[key] = False
        elif key == 'video_view_type':
            st.session_state[key] = "Side View"
        elif key == 'target_height_pct':
            st.session_state[key] = 40.0
        elif key == 'chart_y_min':
            st.session_state[key] = 0
        elif key == 'chart_y_max':
            st.session_state[key] = st.session_state.get('video_height', 1080)
        else:
            st.session_state[key] = None
    
    if rerun:
        st.rerun()

# --- SIDEBAR CONTROLS ---

st.sidebar.markdown("## 🎬 Phase 1: Detection")
st.sidebar.caption("⚠️ Changes require 'Rerun Analysis' button")
st.sidebar.divider()

view_type = st.sidebar.radio("Camera View", ["Side View", "Down-the-Line (DTL)"], index=0, 
                             help="Side View: Camera is perpendicular to the path. DTL: Camera is behind or in front of the trajectory.")
st.sidebar.divider()

sensitivity = st.sidebar.slider("Shape Sensitivity", 0.1, 1.0, 0.6, 0.05)
st.sidebar.caption("How circular a blob must be to count as a ball.")

sat_val = st.sidebar.slider("Yellow Threshold", 50, 255, 120)
st.sidebar.caption("Minimum color saturation to detect as yellow.")

st.sidebar.divider()
st.sidebar.markdown("## 📊 Analysis Phase")
st.sidebar.caption("✨ Updates instantly when adjusted")
st.sidebar.divider()

st.sidebar.markdown("**🎯 Launch Zone**")
launch_time_percentile = st.sidebar.slider("Time Window (%)", 10, 50, 20, 5, key="launch_zone_slider")
st.sidebar.caption("Use earliest X% of trajectories to define launch zone.")

st.sidebar.markdown("**📏 Domain Filters**")
angle_range = st.sidebar.slider("Launch Angle (°)", -45, 120, (20, 80), 5)
min_angle, max_angle = angle_range
st.sidebar.caption(f"Valid: {min_angle}° to {max_angle}°")

# Dynamic height filter based on video resolution
max_v_ht = st.session_state.get('video_height', 1080)
height_range = st.sidebar.slider("Max Height (px from top)", 0, max_v_ht, (0, max_v_ht), 50)
min_height, max_height = height_range
st.sidebar.caption(f"Valid: {min_height} to {max_height} px")

min_velocity = st.sidebar.slider("Min Velocity (px/s)", 0, 100, 10, 5)
st.sidebar.caption(f"Minimum: {min_velocity} px/s")

st.sidebar.divider()
st.sidebar.markdown("**🖼️ Chart View**")
chart_y_range = st.sidebar.slider("Chart Y-Axis Range", 0, max_v_ht, 
                                 (st.session_state.chart_y_min, st.session_state.chart_y_max), 50)
st.session_state.chart_y_min, st.session_state.chart_y_max = chart_y_range
st.sidebar.caption(f"View Window: {st.session_state.chart_y_min} to {st.session_state.chart_y_max} px")

st.sidebar.markdown("**🎯 Target Accuracy**")
enable_accuracy = st.sidebar.checkbox("Show Accuracy Analysis", value=True)

if enable_accuracy:
    # --- Bidirectional Sync Logic ---
    def on_slider_change():
        st.session_state.target_height_pct = st.session_state.target_height_slider
    
    def on_number_change():
        st.session_state.target_height_pct = st.session_state.target_height_number

    col_h1, col_h2 = st.sidebar.columns([2, 1])
    with col_h1:
        st.slider("Target Height (%)", 0.0, 100.0, 
                  key="target_height_slider",
                  value=st.session_state.target_height_pct,
                  step=0.1,
                  on_change=on_slider_change)
    with col_h2:
        st.number_input("Value", 0.0, 100.0, 
                        key="target_height_number",
                        value=st.session_state.target_height_pct,
                        step=0.1,
                        label_visibility="collapsed",
                        on_change=on_number_change)
    
    target_height_pct = st.session_state.target_height_pct
    st.sidebar.caption(f"Landing accuracy at {target_height_pct}% height (descending only)")
else:
    target_height_pct = 40.0

st.sidebar.markdown("**📈 Statistical**")
enable_stats_filtering = st.sidebar.checkbox("IQR Outlier Removal", value=False)
stats_sensitivity = st.sidebar.slider("IQR Threshold", 1.5, 4.0, 2.5, 0.5) if enable_stats_filtering else 2.5
if enable_stats_filtering:
    st.sidebar.caption(f"Threshold: {stats_sensitivity} (1.5=strict, 3.0=lenient)")

# Fixed parameters (good defaults, rarely need tuning)
match_threshold = 80  # pixels
memory_frames = 15  # frames
outlier_sensitivity = 3.0  # launch zone filter sensitivity
speed = 1.0  # playback speed
auto_fps = True
manual_fps = 30

if st.sidebar.button("Reset All Data"):
    reset_app_state()

col_upload, col_rerun_btn = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload your MOV file", type=['mov', 'mp4'])

with col_rerun_btn:
    if st.button("🔄 Rerun Analysis", disabled=not st.session_state.video_path, use_container_width=True):
        reset_app_state()

if uploaded_file:
    temp_path = "temp_video.mov"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.video_path = temp_path
    st.session_state.original_filename = os.path.splitext(uploaded_file.name)[0]
elif st.session_state.video_path:
    temp_path = st.session_state.video_path
else:
    temp_path = None

# Phase indicator (placed here so it knows the current state)
if st.session_state.analysis_complete:
    st.success("📊 **Analysis Phase** - Adjust sliders to refine results")
elif st.session_state.video_path and not st.session_state.analysis_complete:
    if st.session_state.raw_trajectories:
        st.info("🎬 **Phase 1: Video Analysis Complete** - Applying filters...")
    else:
        st.warning("🎬 **Phase 1: Video Analysis Running** - Processing video...")
else:
    st.info("📤 **Ready** - Upload a video to begin analysis")

# Create UI layout (persistent across reruns)
col1, col2 = st.columns([1, 1])
video_feed = col1.empty()
graph_plot = col2.empty()

pause_btn = st.empty()
status_bar = st.empty()
summary_area = st.empty()
distribution_plot_angle = st.empty()  # Dedicated container for angle distribution
distribution_plot_height = st.empty()  # Dedicated container for height distribution
trend_plot = st.empty()
log_table = st.empty()

# Create matplotlib figure for trajectory chart (reused across renders)
fig, ax = plt.subplots(figsize=(6, 5))
colormap = plt.get_cmap('gist_rainbow')

# --- Unified Render Functions (module level, available to both phases) ---

def render_trajectory_chart_unified(all_trajs, live_tracks, ball_log, width_dim, height_dim, show_target=False, target_height_pct=40):
    """Unified trajectory chart renderer for both phases."""
    def build_seq_map(ball_log):
        if not ball_log:
            return {}
        log_sorted = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
        return {entry['Ball #']: i + 1 for i, entry in enumerate(log_sorted)}

    id_to_seq = build_seq_map(ball_log)
    ax.clear()
    ax.set_facecolor('#1e1e1e')

    # Draw target line if accuracy analysis is enabled
    if show_target:
        target_y = int(height_dim * target_height_pct / 100)
        ax.axhline(-target_y, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'Target @ {target_height_pct}%')

    # Draw launch zone if available
    if st.session_state.launch_zone_center is not None:
        center_x, center_y = st.session_state.launch_zone_center
        radius = st.session_state.launch_zone_radius
        circle = plt.Circle((center_x, -center_y), radius,
                           color='cyan', fill=False, linewidth=2,
                           linestyle='--', alpha=0.6, label='Launch Zone')
        ax.add_patch(circle)
        ax.plot(center_x, -center_y, 'x', color='cyan', markersize=10, markeredgewidth=2)

    for track in all_trajs:
        pts = np.array(track['path'])
        ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=1, alpha=0.85)
        # Label at end of trajectory (no box)
        end_idx = -1
        ax.text(pts[end_idx, 0], -pts[end_idx, 1],
                str(id_to_seq.get(track['id'], track['id'])),
                color=track['color'], fontsize=10, fontweight='bold',
                ha='center', va='center')
    for track in live_tracks:
        if len(track['path']) > 1:
            pts = np.array(track['path'])
            ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=2)
            # Label at end for live tracks (no box)
            end_idx = -1
            ax.text(pts[end_idx, 0], -pts[end_idx, 1],
                    str(id_to_seq.get(track['id'], '…')),
                    color=track['color'], fontsize=10, fontweight='bold',
                    ha='center', va='center')

    # Autoscale axes to fit data with padding
    y_min_plot = -st.session_state.chart_y_max
    y_max_plot = -st.session_state.chart_y_min
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.autoscale(enable=True, axis='x', tight=False)
    ax.margins(x=0.05)  # 5% padding around x data
    graph_plot.pyplot(fig)

def render_summary_unified(df, accuracy_data=None):
    """Unified summary renderer for both phases.

    Args:
        df: Ball log dataframe
        accuracy_data: Optional dict from calculate_target_accuracy with intercept_map
    """
    with summary_area.container():
        st.markdown(f"### Balls: {len(df)}")

        # Add target distance column if accuracy data available
        display_df = df.copy()
        if accuracy_data is not None and 'intercept_map' in accuracy_data:
            # Add target distance for each ball
            target_distances = []
            target_positions = []
            extrapolated_flags = []

            for _, row in display_df.iterrows():
                ball_id = row.get('Ball #', None)  # May have been dropped
                if ball_id and ball_id in accuracy_data['intercept_map']:
                    info = accuracy_data['intercept_map'][ball_id]
                    target_distances.append(round(info['target_distance'], 1))
                    target_positions.append(round(info['target_x'], 1))
                    extrapolated_flags.append('*' if info['extrapolated'] else '')
                else:
                    target_distances.append(None)
                    target_positions.append(None)
                    extrapolated_flags.append('')

            display_df['Target X (px)'] = target_positions
            display_df['Target Distance (px)'] = target_distances
            display_df['Extrap'] = extrapolated_flags

        # Drop Ball # column (internal ID, not useful to user)
        if 'Ball #' in display_df.columns:
            display_df = display_df.drop(columns=['Ball #'])

        # Update stats to include target distance if available
        metrics = ["Velocity (px/s)", "Launch Angle (°)", "Max Height (px from top)"]
        avgs = [
            f"{df['Initial Velocity (px/s)'].mean():.0f}",
            f"{df['Launch Angle (°)'].mean():.1f}",
            f"{df['Max Height (px from top)'].mean():.0f}",
        ]
        stds = [
            f"{df['Initial Velocity (px/s)'].std():.0f}",
            f"{df['Launch Angle (°)'].std():.1f}",
            f"{df['Max Height (px from top)'].std():.0f}",
        ]

        if 'Target Distance (px)' in display_df.columns:
            target_dists = display_df['Target Distance (px)'].dropna()
            if len(target_dists) > 0:
                metrics.append("Target Distance (px)")
                avgs.append(f"{target_dists.mean():.0f}")
                stds.append(f"{target_dists.std():.0f}")

        stats = pd.DataFrame({
            "Metric": metrics,
            "Avg": avgs,
            "Std Dev": stds,
        })
        st.dataframe(stats, hide_index=True, use_container_width=True)

    # Add distribution plots for launch angle and max height
    if len(df) >= 3:
        # Launch Angle Distribution
        fig_dist_angle, ax_dist_angle = plt.subplots(figsize=(8, 3))
        fig_dist_angle.patch.set_facecolor('#1e1e1e')
        ax_dist_angle.set_facecolor('#1e1e1e')
        ax_dist_angle.tick_params(colors='white')
        ax_dist_angle.spines[:].set_color('#444')

        angles = df['Launch Angle (°)']
        ax_dist_angle.hist(angles, bins=min(20, len(df)), color='#ff7f0e', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_dist_angle.axvline(angles.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {angles.mean():.1f}°')
        ax_dist_angle.axvline(angles.median(), color='cyan', linestyle='--', linewidth=2, label=f'Median: {angles.median():.1f}°')
        ax_dist_angle.set_xlabel('Launch Angle (°)', color='white', fontsize=9)
        ax_dist_angle.set_ylabel('Count', color='white', fontsize=9)
        ax_dist_angle.set_title('Launch Angle Distribution', color='white', fontsize=10)
        ax_dist_angle.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444', labelcolor='white')
        ax_dist_angle.grid(True, alpha=0.2, color='white')
        distribution_plot_angle.pyplot(fig_dist_angle)
        plt.close(fig_dist_angle)

        # Max Height Distribution
        fig_dist_height, ax_dist_height = plt.subplots(figsize=(8, 3))
        fig_dist_height.patch.set_facecolor('#1e1e1e')
        ax_dist_height.set_facecolor('#1e1e1e')
        ax_dist_height.tick_params(colors='white')
        ax_dist_height.spines[:].set_color('#444')

        heights = df['Max Height (px from top)']
        ax_dist_height.hist(heights, bins=min(20, len(df)), color='#2ecc71', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_dist_height.axvline(heights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {heights.mean():.0f}px')
        ax_dist_height.axvline(heights.median(), color='cyan', linestyle='--', linewidth=2, label=f'Median: {heights.median():.0f}px')
        ax_dist_height.set_xlabel('Max Height (px from top)', color='white', fontsize=9)
        ax_dist_height.set_ylabel('Count', color='white', fontsize=9)
        ax_dist_height.set_title('Max Height Distribution', color='white', fontsize=10)
        ax_dist_height.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444', labelcolor='white')
        ax_dist_height.grid(True, alpha=0.2, color='white')
        distribution_plot_height.pyplot(fig_dist_height)
        plt.close(fig_dist_height)
    else:
        distribution_plot_angle.empty()  # Clear if not enough data
        distribution_plot_height.empty()

    # Trend charts over time
    if len(df) >= 2:
        fig_trend, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        fig_trend.patch.set_facecolor('#1e1e1e')
        x = df["Launch Time (s)"]
        ball_numbers = df["Ball"]
        for sub_ax in (ax1, ax2, ax3):
            sub_ax.set_facecolor('#1e1e1e')
            sub_ax.tick_params(colors='white')
            sub_ax.spines[:].set_color('#444')
        # Velocity plot with ball number labels
        ax1.plot(x, df["Initial Velocity (px/s)"], 'o-', color='#00bfff')
        ax1.axhline(df["Initial Velocity (px/s)"].mean(), color='#00bfff', linestyle='--', alpha=0.4)
        ax1.set_ylabel("Init Velocity\n(px/s)", color='white', fontsize=8)
        # Add ball number labels to velocity plot
        for i, (time, vel, ball_num) in enumerate(zip(x, df["Initial Velocity (px/s)"], ball_numbers)):
            ax1.annotate(str(ball_num), (time, vel), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=7, color='#00bfff', alpha=0.7)
        ax2.plot(x, df["Launch Angle (°)"], 'o-', color='#ff7f0e')
        ax2.axhline(df["Launch Angle (°)"].mean(), color='#ff7f0e', linestyle='--', alpha=0.4)
        ax2.set_ylabel("Launch Angle\n(°)", color='white', fontsize=8)
        ax3.plot(x, df["Max Height (px from top)"], 'o-', color='#2ecc71')
        ax3.axhline(df["Max Height (px from top)"].mean(), color='#2ecc71', linestyle='--', alpha=0.4)
        ax3.set_ylabel("Max Height\n(px from top)", color='white', fontsize=8)
        ax3.set_xlabel("Time (s)", color='white', fontsize=9)
        fig_trend.tight_layout()
        trend_plot.pyplot(fig_trend)
        plt.close(fig_trend)

    log_table.dataframe(display_df, hide_index=True, use_container_width=True)

def calculate_projected_accuracy(trajectories, ball_log, target_y_px, frame_height):
    """Project trajectories using reliable early data (launch to apex) and physics.
    
    Fits a parabola using GRAVITY_ACCEL to predict landing position.
    """
    projected_intercepts = []
    
    # Create mapping of Ball # to sequential number
    sorted_log = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
    ball_id_to_seq = {entry['Ball #']: i + 1 for i, entry in enumerate(sorted_log)}

    for track, log_entry in zip(trajectories, ball_log):
        path = track['path']
        if len(path) < 5: continue
        
        ball_num = ball_id_to_seq.get(track['id'], '?')
        pts = np.array(path)
        
        # 1. Identify Reliable Segment (Launch to Apex + 3 frames)
        # Minimum Y is the apex
        apex_idx = np.argmin(pts[:, 1])
        # Use data from start up to slightly past apex
        reliable_end = min(len(pts), apex_idx + 4)
        reliable_pts = pts[:reliable_end]
        
        if len(reliable_pts) < 3: continue
        
        # 2. Physics Model Fitting
        # y(t) = y0 + vy0*t + 0.5*g*t^2
        # x(t) = x0 + vx*t
        # We know g = GRAVITY_ACCEL. We solve for vx and vy0.
        
        # Time steps (frames)
        t = np.arange(len(reliable_pts))
        x = reliable_pts[:, 0]
        y = reliable_pts[:, 1]
        
        # Fit X (linear)
        vx_fit = np.polyfit(t, x, 1)[0]
        x0_fit = x[0]
        
        # Fit Y (quadratic with fixed gravity)
        # y - 0.5*g*t^2 = y0 + vy0*t
        y_adj = y - 0.5 * GRAVITY_ACCEL * (t**2)
        vy0_fit, y0_fit = np.polyfit(t, y_adj, 1)
        
        # 3. Project to Target Height
        # Solve for t: 0.5*g*t^2 + vy0*t + (y0 - target_y) = 0
        a = 0.5 * GRAVITY_ACCEL
        b = vy0_fit
        c = y0_fit - target_y_px
        
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            # We want the positive root (descending)
            t_intercept = (-b + np.sqrt(discriminant)) / (2*a)
            
            # Final projected X
            x_projected = x0_fit + vx_fit * t_intercept
            
            projected_intercepts.append({
                'ball_id': track['id'],
                'ball_num': ball_num,
                'x': x_projected,
                'y': target_y_px,
                'launch_time': log_entry['Launch Time (s)'],
                'extrapolated': True,
                'is_projection': True
            })

    if not projected_intercepts: return None

    # Calculate metrics
    x_positions = np.array([i['x'] for i in projected_intercepts])
    mean_x = np.mean(x_positions)
    std_x = np.std(x_positions)
    distances_from_mean = np.abs(x_positions - mean_x)
    
    return {
        'intercepts': projected_intercepts,
        'mean_x': mean_x,
        'std_x': std_x,
        'cep': np.median(distances_from_mean),
        'r95': np.percentile(distances_from_mean, 95),
        'min_x': np.min(x_positions),
        'max_x': np.max(x_positions),
        'spread': np.max(x_positions) - np.min(x_positions),
        'intercept_map': {i['ball_id']: {'target_x': i['x'], 'target_distance': abs(i['x'] - mean_x), 'extrapolated': True} for i in projected_intercepts}
    }

def calculate_target_accuracy(trajectories, ball_log, target_y_px, frame_height):
    """Calculate where each trajectory crosses a target height on DESCENDING path (or extrapolate).

    Returns dict with accuracy metrics AND intercept_map for joining with ball_log.
    """
    intercepts = []

    # Create mapping of Ball # to sequential number (sorted by time)
    sorted_log = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
    ball_id_to_seq = {entry['Ball #']: i + 1 for i, entry in enumerate(sorted_log)}

    # Safety check: ensure trajectories and ball_log are synchronized
    if len(trajectories) != len(ball_log):
        st.warning(f"⚠️ Data mismatch: {len(trajectories)} trajectories but {len(ball_log)} log entries")
        return None

    for track, log_entry in zip(trajectories, ball_log):
        path = track['path']
        ball_num = ball_id_to_seq.get(track['id'], '?')

        # Find peak (highest point = minimum Y value)
        if len(path) < 2:
            continue  # Skip invalid trajectories
        y_values = [pt[1] for pt in path]
        if not y_values:  # Safety check for empty list
            continue
        peak_idx = y_values.index(min(y_values))

        # Look for intercept AFTER peak (descending portion)
        descending_path = path[peak_idx:]
        found_intercept = False

        for i in range(len(descending_path) - 1):
            y1, y2 = descending_path[i][1], descending_path[i+1][1]
            # Check if descending (y increasing) and crosses target
            if y1 < target_y_px <= y2:
                # Linear interpolation to find exact X position
                t = (target_y_px - y1) / (y2 - y1)
                x_intercept = descending_path[i][0] + t * (descending_path[i+1][0] - descending_path[i][0])
                intercepts.append({
                    'ball_id': track['id'],
                    'ball_num': ball_num,
                    'x': x_intercept,
                    'y': target_y_px,
                    'launch_time': log_entry['Launch Time (s)'],
                    'extrapolated': False
                })
                found_intercept = True
                break

        # If no intercept found and trajectory ended early, extrapolate
        if not found_intercept and len(descending_path) >= 3:
            last_y = descending_path[-1][1]
            # Only extrapolate if trajectory ended above target (hasn't reached it yet)
            if last_y < target_y_px:
                # Use last 3 points to estimate velocity
                p1, p2, p3 = descending_path[-3], descending_path[-2], descending_path[-1]
                vx = (p3[0] - p1[0]) / 2  # average horizontal velocity
                vy = (p3[1] - p1[1]) / 2  # average vertical velocity (positive = falling)

                # Extrapolate: how many steps to reach target?
                if vy > 0.5:  # Must be descending
                    steps_needed = (target_y_px - p3[1]) / vy
                    x_extrapolated = p3[0] + vx * steps_needed

                    # Sanity check: don't extrapolate too far
                    if 0 < steps_needed < 50:  # reasonable extrapolation range
                        intercepts.append({
                            'ball_id': track['id'],
                            'ball_num': ball_num,
                            'x': x_extrapolated,
                            'y': target_y_px,
                            'launch_time': log_entry['Launch Time (s)'],
                            'extrapolated': True
                        })
                        found_intercept = True

    if not intercepts:
        return None

    # Calculate accuracy metrics
    x_positions = np.array([i['x'] for i in intercepts])

    # Check for NaN values in positions
    if np.any(np.isnan(x_positions)):
        st.warning("⚠️ Invalid intercept calculations (NaN values detected)")
        return None

    mean_x = np.mean(x_positions)
    std_x = np.std(x_positions)

    # CEP (Circular Error Probable) - radius containing 50% of shots
    distances_from_mean = np.abs(x_positions - mean_x)
    cep = np.median(distances_from_mean)

    # R95 - radius containing 95% of shots
    r95 = np.percentile(distances_from_mean, 95)

    # Create mapping of ball_id to target info for table augmentation
    intercept_map = {}
    for intercept in intercepts:
        dist_from_mean = abs(intercept['x'] - mean_x)
        intercept_map[intercept['ball_id']] = {
            'target_x': intercept['x'],
            'target_distance': dist_from_mean,
            'extrapolated': intercept.get('extrapolated', False)
        }

    return {
        'intercepts': intercepts,
        'mean_x': mean_x,
        'std_x': std_x,
        'cep': cep,
        'r95': r95,
        'min_x': np.min(x_positions),
        'max_x': np.max(x_positions),
        'spread': np.max(x_positions) - np.min(x_positions),
        'intercept_map': intercept_map
    }

def generate_pdf_report(trajectories, ball_log, accuracy_data, filter_stats, width, height, target_height_pct, output_path, snapshots=None):
    """Generate comprehensive PDF report with all visualizations and statistics."""
    if snapshots is None:
        snapshots = []
    with PdfPages(output_path) as pdf:
        # Page 1: Detection Overview (Chart + Snapshots)
        fig_overview = plt.figure(figsize=(11, 8.5))
        fig_overview.patch.set_facecolor('white')

        # Top: Trajectory Chart (occupies top ~45%)
        ax_traj = fig_overview.add_axes([0.1, 0.55, 0.8, 0.35])  # [left, bottom, width, height]
        ax_traj.set_facecolor('white')

        # Build sequence map
        sorted_log = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
        ball_id_to_seq = {entry['Ball #']: i + 1 for i, entry in enumerate(sorted_log)}

        # Draw trajectories
        for track in trajectories:
            pts = np.array(track['path'])
            ax_traj.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=1.5, alpha=0.9)
            end_idx = -1
            ax_traj.text(pts[end_idx, 0], -pts[end_idx, 1],
                        str(ball_id_to_seq.get(track['id'], track['id'])),
                        color=track['color'], fontsize=10, fontweight='bold',
                        ha='center', va='center')

        # Draw launch zone
        if filter_stats.get('launch_zone_info'):
            lz = filter_stats['launch_zone_info']
            circle = plt.Circle((lz['center'][0], -lz['center'][1]), lz['radius'],
                               color='darkcyan', fill=False, linewidth=2, linestyle='--', alpha=0.8)
            ax_traj.add_patch(circle)
            ax_traj.plot(lz['center'][0], -lz['center'][1], 'x', color='darkcyan', markersize=10, markeredgewidth=2)

        ax_traj.autoscale(enable=True, axis='both', tight=False)
        ax_traj.margins(0.05)
        
        view_label = st.session_state.get('video_view_type', 'Side View')
        x_label = 'Lateral Position (px)' if view_label == 'Down-the-Line (DTL)' else 'X Position (px)'
        ax_traj.set_xlabel(x_label, color='black', fontsize=10)
        ax_traj.set_ylabel('Y Position (px)', color='black', fontsize=10)
        ax_traj.set_title(f'Ball Trajectories & Analysis ({view_label})', color='black', fontsize=12, fontweight='bold')
        ax_traj.tick_params(colors='black', labelsize=8)
        for spine in ax_traj.spines.values():
            spine.set_color('#444')
        ax_traj.grid(True, alpha=0.3, color='grey')

        # Bottom: 6 Snapshots in a 2x3 grid (occupies bottom ~45%)
        if snapshots:
            for i, snap in enumerate(snapshots[:6]):
                row = 1 - (i // 3)  # 0 or 1, inverted for bottom-up axes coords
                col = i % 3         # 0, 1, or 2
                
                # Axes position: [left, bottom, width, height]
                ax_snap = fig_overview.add_axes([0.05 + col*0.31, 0.05 + row*0.22, 0.28, 0.20])
                ax_snap.imshow(snap)
                ax_snap.axis('off')
                ax_snap.set_title(f"Detection {i+1}", fontsize=8, color='black')

        pdf.savefig(fig_overview, facecolor='white')
        plt.close(fig_overview)

        # Page 2: Summary Statistics and Distributions
        fig_stats = plt.figure(figsize=(11, 8.5))
        fig_stats.patch.set_facecolor('white')

        # Summary text
        ax_summary = fig_stats.add_subplot(4, 1, 1)
        ax_summary.axis('off')
        df = pd.DataFrame(ball_log)
        df = df.sort_values("Launch Time (s)").reset_index(drop=True)

        summary_text = f"Ball Tracking Analysis Report\n\n"
        summary_text += f"Total Balls Analyzed: {len(df)}\n"
        summary_text += f"Camera View: {st.session_state.get('video_view_type', 'Side View')}\n"
        summary_text += f"Initial Detections: {filter_stats['initial']}\n"
        summary_text += f"Filtered Out: {filter_stats['spatial_removed']} spatial, "
        summary_text += f"{filter_stats['domain_removed']} domain, "
        summary_text += f"{filter_stats['stats_removed']} statistical\n\n"
        summary_text += f"Velocity: {df['Initial Velocity (px/s)'].mean():.0f} ± {df['Initial Velocity (px/s)'].std():.0f} px/s\n"
        summary_text += f"Launch Angle: {df['Launch Angle (°)'].mean():.1f} ± {df['Launch Angle (°)'].std():.1f}°\n"
        summary_text += f"Max Height: {df['Max Height (px from top)'].mean():.0f} ± {df['Max Height (px from top)'].std():.0f} px"

        ax_summary.text(0.05, 0.5, summary_text, fontsize=12, color='black',
                       verticalalignment='center', family='monospace')

        # Angle distribution
        ax_angle = fig_stats.add_subplot(4, 1, 2)
        ax_angle.set_facecolor('white')
        angles = df['Launch Angle (°)']
        ax_angle.hist(angles, bins=min(20, len(df)), color='#ff7f0e', alpha=0.7, edgecolor='black')
        ax_angle.axvline(angles.mean(), color='red', linestyle='--', linewidth=2)
        ax_angle.set_xlabel('Launch Angle (°)', color='black')
        ax_angle.set_ylabel('Count', color='black')
        ax_angle.set_title('Launch Angle Distribution', color='black', fontweight='bold')
        ax_angle.tick_params(colors='black')
        for spine in ax_angle.spines.values():
            spine.set_color('#444')

        # Height distribution
        ax_height = fig_stats.add_subplot(4, 1, 3)
        ax_height.set_facecolor('white')
        heights = df['Max Height (px from top)']
        ax_height.hist(heights, bins=min(20, len(df)), color='#2ecc71', alpha=0.7, edgecolor='black')
        ax_height.axvline(heights.mean(), color='red', linestyle='--', linewidth=2)
        ax_height.set_xlabel('Max Height (px)', color='black')
        ax_height.set_ylabel('Count', color='black')
        ax_height.set_title('Max Height Distribution', color='black', fontweight='bold')
        ax_height.tick_params(colors='black')
        for spine in ax_height.spines.values():
            spine.set_color('#444')

        # Velocity distribution
        ax_vel = fig_stats.add_subplot(4, 1, 4)
        ax_vel.set_facecolor('white')
        vels = df['Initial Velocity (px/s)']
        ax_vel.hist(vels, bins=min(20, len(df)), color='#00bfff', alpha=0.7, edgecolor='black')
        ax_vel.axvline(vels.mean(), color='red', linestyle='--', linewidth=2)
        ax_vel.set_xlabel('Initial Velocity (px/s)', color='black')
        ax_vel.set_ylabel('Count', color='black')
        ax_vel.set_title('Velocity Distribution', color='black', fontweight='bold')
        ax_vel.tick_params(colors='black')
        for spine in ax_vel.spines.values():
            spine.set_color('#444')

        fig_stats.tight_layout()
        pdf.savefig(fig_stats, facecolor='white')
        plt.close(fig_stats)

        # Page 3: Trend Charts
        if len(df) >= 2:
            fig_trend = plt.figure(figsize=(11, 8.5))
            fig_trend.patch.set_facecolor('white')

            x = df["Launch Time (s)"]

            ax1 = fig_trend.add_subplot(3, 1, 1)
            ax1.set_facecolor('white')
            ax1.plot(x, df["Initial Velocity (px/s)"], 'o-', color='#00bfff', linewidth=2, markersize=6)
            ax1.axhline(df["Initial Velocity (px/s)"].mean(), color='#00bfff', linestyle='--', alpha=0.4, linewidth=2)
            ax1.set_ylabel("Velocity (px/s)", color='black', fontsize=11)
            ax1.tick_params(colors='black')
            ax1.set_title('Performance Trends Over Time', color='black', fontsize=14, fontweight='bold')
            for spine in ax1.spines.values():
                spine.set_color('#444')
            ax1.grid(True, alpha=0.3, color='grey')

            ax2 = fig_trend.add_subplot(3, 1, 2)
            ax2.set_facecolor('white')
            ax2.plot(x, df["Launch Angle (°)"], 'o-', color='#ff7f0e', linewidth=2, markersize=6)
            ax2.axhline(df["Launch Angle (°)"].mean(), color='#ff7f0e', linestyle='--', alpha=0.4, linewidth=2)
            ax2.set_ylabel("Angle (°)", color='black', fontsize=11)
            ax2.tick_params(colors='black')
            for spine in ax2.spines.values():
                spine.set_color('#444')
            ax2.grid(True, alpha=0.3, color='grey')

            ax3 = fig_trend.add_subplot(3, 1, 3)
            ax3.set_facecolor('white')
            ax3.plot(x, df["Max Height (px from top)"], 'o-', color='#2ecc71', linewidth=2, markersize=6)
            ax3.axhline(df["Max Height (px from top)"].mean(), color='#2ecc71', linestyle='--', alpha=0.4, linewidth=2)
            ax3.set_ylabel("Height (px)", color='black', fontsize=11)
            ax3.set_xlabel("Time (s)", color='black', fontsize=11)
            ax3.tick_params(colors='black')
            for spine in ax3.spines.values():
                spine.set_color('#444')
            ax3.grid(True, alpha=0.3, color='grey')

            fig_trend.tight_layout()
            pdf.savefig(fig_trend, facecolor='white')
            plt.close(fig_trend)

        # Page 4: Target Accuracy (if available)
        if accuracy_data is not None:
            fig_acc = plt.figure(figsize=(11, 8.5))
            fig_acc.patch.set_facecolor('white')

            # Top: Trajectories with target line
            ax_top = fig_acc.add_subplot(2, 1, 1)
            ax_top.set_facecolor('white')

            for track in trajectories:
                pts = np.array(track['path'])
                ax_top.plot(pts[:, 0], pts[:, 1], color=track['color'], linewidth=1, alpha=0.6)

            target_y_px = int(height * target_height_pct / 100)
            ax_top.axhline(target_y_px, color='red', linestyle='--', linewidth=2, label=f'Target @ {target_height_pct}%')

            # Mark intercepts
            for intercept in accuracy_data['intercepts']:
                if intercept.get('extrapolated', False):
                    ax_top.plot(intercept['x'], intercept['y'], 'o', color='none', markersize=10,
                               markeredgecolor='orange', markeredgewidth=2)
                else:
                    ax_top.plot(intercept['x'], intercept['y'], 'o', color='yellow', markersize=8,
                               markeredgecolor='red', markeredgewidth=2)

            # Autoscale x-axis to zoom into trajectory region
            ax_top.autoscale(enable=True, axis='x', tight=False)
            ax_top.margins(x=0.1, y=0)  # 10% padding on x-axis
            ax_top.set_ylim(height, 0)  # Keep full Y range
            ax_top.set_xlabel('X Position (px)', color='black', fontsize=11)
            ax_top.set_ylabel('Y Position (px)', color='black', fontsize=11)
            ax_top.set_title(f'Target Accuracy Analysis - {target_height_pct}% Height', color='black', fontsize=14, fontweight='bold')
            ax_top.tick_params(colors='black')
            for spine in ax_top.spines.values():
                spine.set_color('#444')
            ax_top.grid(True, alpha=0.3, color='grey')
            ax_top.legend(facecolor='white', edgecolor='#444', labelcolor='black')

            # Bottom: Accuracy metrics and scatter
            ax_bottom = fig_acc.add_subplot(2, 1, 2)
            ax_bottom.set_facecolor('white')

            x_positions = [i['x'] for i in accuracy_data['intercepts']]
            times = [i['launch_time'] for i in accuracy_data['intercepts']]

            ax_bottom.scatter(times, x_positions, c='cyan', s=100, alpha=0.8, edgecolors='black')
            ax_bottom.axhline(accuracy_data['mean_x'], color='green', linestyle='-', linewidth=2)
            ax_bottom.axhline(accuracy_data['mean_x'] + accuracy_data['std_x'], color='orange', linestyle='--', alpha=0.6)
            ax_bottom.axhline(accuracy_data['mean_x'] - accuracy_data['std_x'], color='orange', linestyle='--', alpha=0.6)

            ax_bottom.set_xlabel('Launch Time (s)', color='black', fontsize=11)
            ax_bottom.set_ylabel('X Position at Target (px)', color='black', fontsize=11)
            ax_bottom.set_title(f'Accuracy: Mean={accuracy_data["mean_x"]:.0f}px, Spread={accuracy_data["std_x"]:.0f}px, CEP={accuracy_data["cep"]:.0f}px',
                               color='black', fontsize=12)
            ax_bottom.tick_params(colors='black')
            for spine in ax_bottom.spines.values():
                spine.set_color('#444')
            ax_bottom.grid(True, alpha=0.3, color='grey')

            fig_acc.tight_layout()
            pdf.savefig(fig_acc, facecolor='white')
            plt.close(fig_acc)

            # Page 5: Correlation Analysis (if available)
            if len(accuracy_data['intercepts']) >= 3:
                # Build correlation dataframe
                corr_data = []
                for intercept in accuracy_data['intercepts']:
                    ball_id = intercept['ball_id']
                    for log_entry in ball_log:
                        if log_entry['Ball #'] == ball_id:
                            corr_data.append({
                                'Target X': intercept['x'],
                                'Velocity': log_entry['Initial Velocity (px/s)'],
                                'Angle': log_entry['Launch Angle (°)'],
                                'Max Height': log_entry['Max Height (px from top)'],
                                'Launch Time': log_entry['Launch Time (s)']
                            })
                            break
                
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    correlations = corr_df.corr()['Target X'].drop('Target X')
                    
                    fig_corr, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
                    fig_corr.patch.set_facecolor('white')
                    
                    metrics = [
                        ('Velocity', 'Velocity', ax1, '#00bfff', 'Initial Velocity (px/s)'),
                        ('Angle', 'Angle', ax2, '#ff7f0e', 'Launch Angle (°)'),
                        ('Max Height', 'Max Height', ax3, '#2ecc71', 'Max Height (px from top)'),
                        ('Launch Time', 'Launch Time', ax4, '#9467bd', 'Launch Time (s)')
                    ]
                    
                    for metric_name, col_name, ax, color, xlabel in metrics:
                        ax.set_facecolor('white')
                        ax.tick_params(colors='black')
                        for spine in ax.spines.values():
                            spine.set_color('#444')
                        
                        x_data = corr_df[col_name]
                        y_data = corr_df['Target X']
                        # Map Ball # to sequential numbers for PDF labeling
                        sorted_log_pdf = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
                        ball_id_to_seq_pdf = {entry['Ball #']: i + 1 for i, entry in enumerate(sorted_log_pdf)}
                        ball_nums_pdf = [ball_id_to_seq_pdf.get(i['ball_id'], '?') for i in accuracy_data['intercepts']]
                        
                        corr_val = correlations.get(metric_name, 0)
                        
                        ax.scatter(x_data, y_data, c=color, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
                        
                        # Add ball number labels to scatter points in PDF
                        for i, txt in enumerate(ball_nums_pdf):
                            ax.annotate(str(txt), (x_data.iloc[i], y_data.iloc[i]), 
                                        textcoords="offset points", xytext=(0, 5), 
                                        ha='center', fontsize=6, color='black', alpha=0.7)

                        # Add trend line
                        if not np.isnan(corr_val) and abs(corr_val) > 0.01:
                            try:
                                z = np.polyfit(x_data, y_data, 1)
                                p = np.poly1d(z)
                                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                                ax.plot(x_line, p(x_line), '--', color='red', linewidth=1.5, alpha=0.6)
                            except: pass
                            
                        ax.set_xlabel(xlabel, color='black', fontsize=10)
                        ax.set_ylabel('Target X (px)', color='black', fontsize=10)
                        ax.set_title(f'{metric_name} Correlation: {corr_val:.3f}', color='black', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3, color='grey')
                    
                    fig_corr.tight_layout(pad=3.0)
                    pdf.savefig(fig_corr, facecolor='white')
                    plt.close(fig_corr)

        # PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Ball Tracking Analysis Report'
        d['Author'] = 'Ball Tracker Dashboard'
        d['Subject'] = 'Trajectory Analysis and Accuracy Metrics'
        d['CreationDate'] = time.strftime('%Y%m%d%H%M%S')

def render_accuracy_analysis(accuracy_data, trajectories, ball_log, target_height_pct, frame_width, frame_height, mode="Actual"):
    """Render target accuracy visualization using pre-calculated accuracy data."""
    if accuracy_data is None:
        target_y_px = int(frame_height * target_height_pct / 100)
        st.warning(f"⚠️ No valid {mode.lower()} data for target height ({target_height_pct}% = {target_y_px}px from top)")
        return

    accuracy = accuracy_data
    target_y_px = int(frame_height * target_height_pct / 100)
    st.markdown(f"### 🎯 Target Accuracy Analysis ({mode})")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Mean X ({mode})", f"{accuracy['mean_x']:.0f} px")
    with col2:
        st.metric(f"Spread ({mode})", f"{accuracy['std_x']:.0f} px")
    with col3:
        st.metric("CEP (50%)", f"{accuracy['cep']:.0f} px")
    with col4:
        st.metric("R95 (95%)", f"{accuracy['r95']:.0f} px")

    # Visualization
    if mode == "Actual":
        st.caption("💡 **Tip**: Click anywhere on the top chart to set the target height interactively!")
    
    # 1. Plotly Interactive Chart (Top)
    fig_top = go.Figure()
    
    # Invisible grid of points to capture clicks anywhere (scaled to current view)
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, frame_width, 20),
        np.linspace(st.session_state.chart_y_min, st.session_state.chart_y_max, 20)
    )
    fig_top.add_trace(go.Scatter(
        x=grid_x.flatten(),
        y=grid_y.flatten(),
        mode='markers',
        marker=dict(color="rgba(255,255,255,0.01)", size=20),
        hoverinfo='none',
        showlegend=False,
        name="bg_click_grid"
    ))

    # Build seq map for labels
    sorted_log = sorted(ball_log, key=lambda x: x['Launch Time (s)'])
    ball_id_to_seq = {entry['Ball #']: i + 1 for i, entry in enumerate(sorted_log)}

    for track in trajectories:
        pts = np.array(track['path'])
        ball_num = ball_id_to_seq.get(track.get('id'), '?')
        color_hex = '#%02x%02x%02x' % (int(track['color'][0]*255), int(track['color'][1]*255), int(track['color'][2]*255))
        
        # Path trace (slightly wider for better clickability)
        fig_top.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode='lines',
            line=dict(color=color_hex, width=3.0, dash='solid' if mode=="Actual" else 'dot'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Peak (Apex) marker and label
        peak_idx = np.argmin(pts[:, 1])
        fig_top.add_trace(go.Scatter(
            x=[pts[peak_idx, 0]], y=[pts[peak_idx, 1]],
            mode='markers+text',
            marker=dict(symbol='x', color=color_hex, size=8),
            text=[str(ball_num)],
            textposition="top center",
            textfont=dict(color=color_hex, size=10),
            name=f"Ball {ball_num}",
            hoverinfo='text'
        ))

    # Intercepts
    for intercept in accuracy['intercepts']:
        ball_num = intercept['ball_num']
        is_extrap = intercept.get('extrapolated', False)
        fig_top.add_trace(go.Scatter(
            x=[intercept['x']], y=[intercept['y']],
            mode='markers',
            marker=dict(
                symbol='circle' if not is_extrap else 'circle-open',
                color='yellow' if mode=="Actual" else '#00ff00', size=10, 
                line=dict(color='red' if mode=="Actual" else '#00ff00', width=2)
            ),
            name=f"{mode} Intercept {ball_num}",
            hoverinfo='name'
        ))

    # Target line (Red Dashed)
    fig_top.add_shape(
        type="line", x0=0, x1=frame_width, y0=target_y_px, y1=target_y_px,
        line=dict(color="Red", width=2, dash="dash")
    )
    fig_top.add_annotation(
        x=frame_width*0.05, y=target_y_px,
        text=f"Target @ {target_height_pct:.1f}%",
        showarrow=False, yshift=10, font=dict(color="Red", size=12)
    )

    fig_top.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1e1e1e',
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis=dict(title="X Position (px)", gridcolor='#333', fixedrange=True),
        yaxis=dict(title="Y Position (px)", gridcolor='#333', 
                   range=[st.session_state.chart_y_max, st.session_state.chart_y_min], 
                   autorange=False, fixedrange=True),
        title=dict(text=f"Trajectories ({mode})", font=dict(size=14)),
        showlegend=False,
        clickmode='event+select',
        dragmode=False
    )

    # Render and capture clicks
    plotly_key = f"accuracy_plotly_{mode}_{st.session_state.chart_y_min}_{st.session_state.chart_y_max}"
    event_data = st.plotly_chart(fig_top, use_container_width=True, on_select="rerun", key=plotly_key, config={'displayModeBar': False})

    # Handle click interaction to set target height (only in Actual mode)
    if mode == "Actual" and event_data and "selection" in event_data and event_data["selection"]["points"]:
        points = event_data["selection"]["points"]
        new_y = points[0].get("y")
        if new_y is not None:
            new_pct = round((new_y / frame_height) * 100, 1)
            new_pct = max(0.0, min(100.0, new_pct))
            if st.session_state.target_height_pct != new_pct:
                st.session_state.target_height_pct = new_pct
                st.rerun()

    # Bottom Chart remains Matplotlib
    fig_acc_bottom, ax_bottom = plt.subplots(figsize=(10, 3.5))
    fig_acc_bottom.patch.set_facecolor('#1e1e1e')
    ax_bottom.set_facecolor('#1e1e1e')
    ax_bottom.tick_params(colors='white')
    ax_bottom.spines[:].set_color('#444')

    x_positions = [i['x'] for i in accuracy['intercepts']]
    ball_nums = [i['ball_num'] for i in accuracy['intercepts']]
    times = [i['launch_time'] for i in accuracy['intercepts']]
    extrapolated = [i.get('extrapolated', False) for i in accuracy['intercepts']]

    # Scatter plot
    color_main = 'cyan' if mode=="Actual" else '#00ff00'
    for t, x, extrap in zip(times, x_positions, extrapolated):
        ax_bottom.scatter(t, x, c=color_main, s=100, alpha=0.7, edgecolors='white')

    # Label with ball numbers
    for x, t, bn in zip(x_positions, times, ball_nums):
        ax_bottom.annotate(str(bn), (t, x), fontsize=8, color=color_main, ha='center', va='bottom', xytext=(0,5), textcoords='offset points')

    ax_bottom.axhline(accuracy['mean_x'], color='green', linestyle='-', linewidth=2, label=f"Mean: {accuracy['mean_x']:.0f}px")
    ax_bottom.axhline(accuracy['mean_x'] + accuracy['std_x'], color='yellow', linestyle='--', linewidth=1, alpha=0.6, label=f'±1σ: {accuracy['std_x']:.0f}px')
    ax_bottom.axhline(accuracy['mean_x'] - accuracy['std_x'], color='yellow', linestyle='--', linewidth=1, alpha=0.6)

    ax_bottom.set_xlabel('Launch Time (s)', color='white')
    ax_bottom.set_ylabel(f'X Position ({mode})', color='white')
    ax_bottom.set_title(f'{mode} Accuracy Distribution (Spread: {accuracy["spread"]:.0f}px)', color='white')
    ax_bottom.legend(facecolor='#1e1e1e', edgecolor='#444', labelcolor='white', loc='best')
    ax_bottom.grid(True, alpha=0.2, color='white')

    fig_acc_bottom.tight_layout()
    st.pyplot(fig_acc_bottom)
    plt.close(fig_acc_bottom)

    # Correlation analysis (Keep for both)
    if len(accuracy['intercepts']) >= 3:
        st.markdown(f"#### 📊 Correlation Analysis ({mode})")
        
        # Build correlation dataframe
        corr_data = []
        for intercept in accuracy['intercepts']:
            ball_id = intercept['ball_id']
            for log_entry in ball_log:
                if log_entry['Ball #'] == ball_id:
                    corr_data.append({
                        'Target X': intercept['x'],
                        'Velocity': log_entry['Initial Velocity (px/s)'],
                        'Angle': log_entry['Launch Angle (°)'],
                        'Max Height': log_entry['Max Height (px from top)'],
                        'Launch Time': log_entry['Launch Time (s)']
                    })
                    break

        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            correlations = corr_df.corr()['Target X'].drop('Target X').sort_values(ascending=False)

            col_corr1, col_corr2, col_corr3, col_corr4 = st.columns(4)
            with col_corr1: st.metric("Velocity", f"{correlations.get('Velocity', 0):.3f}")
            with col_corr2: st.metric("Angle", f"{correlations.get('Angle', 0):.3f}")
            with col_corr3: st.metric("Max Height", f"{correlations.get('Max Height', 0):.3f}")
            with col_corr4: st.metric("Launch Time", f"{correlations.get('Launch Time', 0):.3f}")

            # Correlation scatter plots
            fig_corr, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig_corr.patch.set_facecolor('#1e1e1e')
            metrics = [
                ('Velocity', 'Velocity', ax1, '#00bfff', 'Initial Velocity (px/s)'),
                ('Angle', 'Angle', ax2, '#ff7f0e', 'Launch Angle (°)'),
                ('Max Height', 'Max Height', ax3, '#2ecc71', 'Max Height (px from top)'),
                ('Launch Time', 'Launch Time', ax4, '#9467bd', 'Launch Time (s)')
            ]
            for metric_name, col_name, ax, color, xlabel in metrics:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values(): spine.set_color('#444')
                x_data, y_data = corr_df[col_name], corr_df['Target X']
                ball_nums_corr = [i['ball_num'] for i in accuracy['intercepts']]
                corr_val = correlations.get(metric_name, 0)
                ax.scatter(x_data, y_data, c=color, s=80, alpha=0.7, edgecolors='white', linewidth=1)
                for i, txt in enumerate(ball_nums_corr):
                    ax.annotate(str(txt), (x_data.iloc[i], y_data.iloc[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='white', alpha=0.8)
                if abs(corr_val) > 0.1 and len(x_data) >= 2:
                    try:
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        ax.plot(np.linspace(x_data.min(), x_data.max(), 100), p(np.linspace(x_data.min(), x_data.max(), 100)), '--', color='red', linewidth=2, alpha=0.6)
                    except (np.linalg.LinAlgError, ValueError):
                        # Polyfit can fail with singular matrices or insufficient data
                        pass
                ax.set_xlabel(xlabel, color='white', fontsize=8)
                ax.set_ylabel('Target X (px)', color='white', fontsize=8)
                ax.set_title(f'{metric_name} Corr: {corr_val:.3f}', color='white', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.2, color='white')
            fig_corr.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

# --- End Unified Render Functions ---

if temp_path and not st.session_state.analysis_complete:
    # PHASE 1: Video Analysis (only run if not already complete)
    cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        st.error("Failed to open video file.")
        st.stop()
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) if auto_fps else manual_fps
        if video_fps <= 0:
            st.warning("⚠️ Video FPS not detected, assuming 30 fps")
            video_fps = 30

        # Store video properties for Phase 2 rendering
        st.session_state.video_width = width
        st.session_state.video_height = height
        st.session_state.video_fps = video_fps
        st.session_state.video_view_type = view_type

        # Adjust tracking parameters based on view type
        is_dtl = (view_type == "Down-the-Line (DTL)")
        # High-speed DTL balls can jump significant pixel distances and appear as streaks
        # Increase threshold further to handle high-velocity launches (e.g. 500+ px jumps)
        effective_match_threshold = match_threshold * (6.0 if is_dtl else 1.0)
        effective_min_ball_area = MIN_BALL_AREA * (0.3 if is_dtl else 1.0) 
        effective_sensitivity = sensitivity * (0.4 if is_dtl else 1.0) 
        effective_memory_frames = memory_frames * (3 if is_dtl else 1) # Allow more missing frames

        lower_yellow = np.array([20, sat_val, 100])
        upper_yellow = np.array([35, 255, 255])

        # --- Helper functions (defined once before the processing loop) ---

        def predict_pos(path):
            if len(path) >= 2:
                dx = path[-1][0] - path[-2][0]
                dy = path[-1][1] - path[-2][1]
                
                # In DTL, pixel velocity decays as it moves away. 
                # Gravity still applies but is less dominant than the Z-axis depth change.
                if is_dtl:
                    dx *= 0.98
                    dy *= 0.98

                # Add gravity to vertical velocity (dy)
                return (path[-1][0] + dx, path[-1][1] + dy + GRAVITY_ACCEL)
            return path[-1]

        def direction_ok(path, new_pos, max_angle=120):
            if len(path) < 2:
                return True
            # Relax direction constraints significantly for DTL to handle jittery detections
            if is_dtl:
                return True # Trust the match_threshold and predict_pos for DTL
                
            vx = path[-1][0] - path[-2][0]
            vy = path[-1][1] - path[-2][1]
            dx = new_pos[0] - path[-1][0]
            dy = new_pos[1] - path[-1][1]

            # Check for zero vectors first
            if (vx == 0 and vy == 0) or (dx == 0 and dy == 0):
                return True

            # Calculate denominator safely to avoid division by zero
            denom = ((vx**2 + vy**2)**0.5 * (dx**2 + dy**2)**0.5)
            if denom < 1e-6:  # Numerical stability threshold
                return True

            cos_a = (vx*dx + vy*dy) / denom
            return np.degrees(np.arccos(np.clip(cos_a, -1, 1))) < max_angle

        def has_bounced(path, consistent_frames=3, min_speed=2):
            """Detect any surface contact by spotting a velocity reversal in x or y.
            Ground bounce: falling (dy>0) then rising (dy<0).
            Wall bounce: x direction reverses.
            Natural arc peak (dy<0 → dy>0) is intentionally ignored.
            """
            if len(path) < consistent_frames + 2:
                return False
            recent = path[-(consistent_frames + 2):]

            def reversed_after_consistent(deltas, was_positive):
                sig = [d for d in deltas if abs(d) >= min_speed]
                if len(sig) < consistent_frames:
                    return False
                prior, last = sig[:-1], sig[-1]
                if was_positive:
                    return sum(d > 0 for d in prior) >= len(prior) - 1 and last < -min_speed
                else:
                    return sum(d < 0 for d in prior) >= len(prior) - 1 and last > min_speed

            dxs = [recent[i+1][0] - recent[i][0] for i in range(len(recent) - 1)]
            dys = [recent[i+1][1] - recent[i][1] for i in range(len(recent) - 1)]
            ground_bounce = reversed_after_consistent(dys, was_positive=True)
            wall_bounce   = (reversed_after_consistent(dxs, was_positive=True) or
                             reversed_after_consistent(dxs, was_positive=False))
            return ground_bounce or wall_bounce

        def finalize(track):
            """Finalize a track and store in RAW data (Phase 1)"""
            p = track['path']

            # Early exit for empty or invalid paths
            if not p or len(p) < 2:
                return

            # Displacement Filter: Ignore "tracks" that haven't moved significantly
            # especially important for DTL where stationary balls on robot are detected
            start_pos = np.array(p[0])
            end_pos = np.array(p[-1])
            total_displacement = np.linalg.norm(end_pos - start_pos)

            # Normalize minimum displacement to frame rate (50px at 30fps baseline)
            min_displacement = 50 * (30 / video_fps)
            if total_displacement < min_displacement:
                return

            n = min(4, len(p))
            init_vel = sum(
                np.linalg.norm(np.array(p[i+1]) - np.array(p[i])) * video_fps
                for i in range(n - 1)
            ) / (n - 1) if n > 1 else 0
            init_vel = float(init_vel)  # Ensure Python float for JSON serialization
            if len(p) >= 2:
                dx = np.mean([p[i+1][0] - p[i][0] for i in range(n - 1)])
                dy = np.mean([p[i+1][1] - p[i][1] for i in range(n - 1)])
                if is_dtl:
                    # In DTL, use abs(dx) to show steepness regardless of horizontal side
                    # Handle near-vertical trajectories without bias
                    if abs(dx) < 0.1:
                        launch_angle = 90.0 if dy < 0 else -90.0
                    else:
                        launch_angle = float(round(np.degrees(np.arctan2(-dy, abs(dx))), 1))
                else:
                    launch_angle = float(round(np.degrees(np.arctan2(-dy, dx)), 1))
            else:
                launch_angle = 0.0
            max_height_px = min(pt[1] for pt in p)
            # Store in RAW data (never modified)
            st.session_state.raw_trajectories.append(track)
            st.session_state.raw_ball_log.append({
                "Ball #": track['id'],
                "Launch Time (s)": round(track['start_time'], 2),
                "Initial Velocity (px/s)": round(init_vel, 1),
                "Launch Angle (°)": launch_angle,
                "Max Height (px from top)": max_height_px,
                "View Type": view_type
            })

        # --- End helpers ---

        active_tracks = st.session_state.active_tracks if st.session_state.pause_frame > 0 else []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to paused position if resuming
        frame_count = st.session_state.pause_frame
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if st.session_state.paused:
            if st.session_state.last_frame is not None:
                video_feed.image(st.session_state.last_frame)
            # Use filtered data if analysis complete, otherwise raw data
            display_trajectories = st.session_state.all_trajectories if st.session_state.analysis_complete else st.session_state.raw_trajectories
            display_log = st.session_state.ball_log if st.session_state.analysis_complete else st.session_state.raw_ball_log
            if display_trajectories or st.session_state.active_tracks:
                render_trajectory_chart_unified(display_trajectories,
                                        st.session_state.active_tracks,
                                        display_log,
                                        width, height, False, 40)
            if display_log:
                df = pd.DataFrame(display_log)
                df = df.sort_values("Launch Time (s)").reset_index(drop=True)
                df.insert(0, "Ball", range(1, len(df) + 1))
                # Don't drop Ball # yet - needed for accuracy mapping
                render_summary_unified(df, None)
            if pause_btn.button("▶ Resume", key="pause_btn"):
                st.session_state.paused = False
                st.rerun()
            status_bar.info(f"Paused at frame {st.session_state.pause_frame}/{total_frames}.")
            cap.release()
            st.stop()
        else:
            if pause_btn.button("⏸ Pause", key="pause_btn"):
                st.session_state.paused = True
                st.session_state.pause_frame = frame_count
                cap.release()
                st.rerun()

        while cap.isOpened():
            start_process_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            current_timestamp = frame_count / video_fps
            t_frame_start = time.time()

            # 1. Image Processing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            # OpenCV version compatibility: cv2.findContours returns different number of values in v3 vs v4
            contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_result[-2]  # Works for both OpenCV 3.x and 4.x

            current_centers = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if area > effective_min_ball_area and perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > effective_sensitivity:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            current_centers.append(center)

            # 2. Tracking & Velocity Calculation
            max_track_frames = int(video_fps * MAX_FLIGHT_TIME_SEC)

            # --- Global Greedy Data Association ---
            # Instead of first-come-first-served, we find the globally best matches
            potential_matches = []
            for c_idx, center in enumerate(current_centers):
                for t_idx, track in enumerate(active_tracks):
                    predicted = predict_pos(track['path'])
                    dist = np.linalg.norm(np.array(center) - np.array(predicted))
                    if dist < effective_match_threshold and direction_ok(track['path'], center):
                        potential_matches.append((dist, c_idx, t_idx))
            
            # Sort matches by distance
            potential_matches.sort(key=lambda x: x[0])
            
            matched_centers = set()
            matched_tracks = set()
            new_active = []
            
            for dist, c_idx, t_idx in potential_matches:
                if c_idx not in matched_centers and t_idx not in matched_tracks:
                    center = current_centers[c_idx]
                    track = active_tracks[t_idx]
                    track['path'].append(center)
                    track['missing_count'] = 0
                    new_active.append(track)
                    matched_centers.add(c_idx)
                    matched_tracks.add(t_idx)
            
            # Create new tracks for unmatched centers
            for c_idx, center in enumerate(current_centers):
                if c_idx not in matched_centers:
                    ball_id = st.session_state.next_ball_id
                    st.session_state.next_ball_id += 1
                    new_active.append({
                        'id': ball_id,
                        'path': [center],
                        'color': colormap((ball_id * 0.618033988749895) % 1.0),
                        'missing_count': 0,
                        'start_time': current_timestamp
                    })
            
            # Handle unmatched (missing) tracks
            for t_idx, track in enumerate(active_tracks):
                if t_idx not in matched_tracks:
                    track['missing_count'] += 1
                    track_age = len(track['path']) + track['missing_count']
                    expired = track_age > max_track_frames
                    lost = track['missing_count'] >= effective_memory_frames
                    if (expired or lost) and len(track['path']) > MIN_TRAJECTORY_LENGTH:
                        finalize(track)
                    elif not expired and not lost:
                        new_active.append(track)
            
            # --------------------------------------
            
            # Finalize tracks whose ballistic arc was interrupted by a bounce
            still_flying = []
            for track in new_active:
                if has_bounced(track['path']) and len(track['path']) > MIN_TRAJECTORY_LENGTH:
                    finalize(track)
                else:
                    still_flying.append(track)

            new_active = still_flying

            active_tracks = new_active
            st.session_state.active_tracks = active_tracks
            st.session_state.pause_frame = frame_count

            # Draw tracking boxes with ball numbers (Only for MOVING tracks)
            for track in active_tracks:
                if len(track['path']) < 2: continue
                
                # Stationary filter for LIVE display: 
                # Don't show boxes for things that haven't moved yet (like balls on robot)
                start_p = np.array(track['path'][0])
                curr_p = np.array(track['path'][-1])
                if np.linalg.norm(curr_p - start_p) < 30:
                    continue

                center = track['path'][-1]
                r, g, b, _ = track['color']
                color_bgr = (int(b * 255), int(g * 255), int(r * 255))
                cv2.circle(frame, center, 15, color_bgr, 2)
                label = str(track['id'])
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, label, (center[0] - tw // 2, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            # --- Snapshots Capture Logic ---
            # Capture 6 snapshots at different points in the video (where balls exist)
            snapshot_interval = max(1, total_frames // 10)
            if len(st.session_state.detection_snapshots) < 6:
                if frame_count % snapshot_interval == 0 and len(active_tracks) > 0:
                    st.session_state.detection_snapshots.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # -------------------------------

            # 3. UI Updates
            finalized_count = len(st.session_state.raw_trajectories)
            cv2.putText(frame, f"Time: {current_timestamp:.2f}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"Balls: {finalized_count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            progress = frame_count / total_frames if total_frames > 0 else 0
            status_bar.progress(progress, text=f"Analyzing... frame {frame_count}/{total_frames} ({progress*100:.1f}%)")

            t_cv = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = rgb_frame
            video_feed.image(rgb_frame)
            t_video = time.time()

            t_graph_start = time.time()
            if frame_count % GRAPH_UPDATE_INTERVAL == 0:
                render_trajectory_chart_unified(st.session_state.raw_trajectories, active_tracks, st.session_state.raw_ball_log, width, height, False, 40)
                if st.session_state.raw_ball_log:
                    df = pd.DataFrame(st.session_state.raw_ball_log)
                    df = df.sort_values("Launch Time (s)").reset_index(drop=True)
                    df.insert(0, "Ball", range(1, len(df) + 1))
                    # Don't drop Ball # yet - needed for accuracy mapping
                    render_summary_unified(df, None)
            t_graph_end = time.time()

            t_total = time.time() - t_frame_start
            print(f"[frame {frame_count:04d}] total={t_total*1000:.1f}ms | cv={( t_cv - t_frame_start)*1000:.1f}ms | video_feed={(t_video - t_cv)*1000:.1f}ms | graph={(t_graph_end - t_graph_start)*1000:.1f}ms | balls={len(st.session_state.raw_trajectories)}")

            # REMOVED: Sync Playback (speed) sleep to ensure frame-by-frame processing is thorough
            # delay = max(0, (1.0 / (video_fps * speed)) - elapsed)
            # time.sleep(delay)

        cap.release()

        # Finalize any tracks still in flight when the video ended
        for track in active_tracks:
            if len(track['path']) > MIN_TRAJECTORY_LENGTH:
                finalize(track)
        active_tracks = []
        st.session_state.active_tracks = []

        # Mark analysis as complete
        st.session_state.analysis_complete = True
        st.rerun()

# PHASE 2: Interactive Filtering Function (callable anytime)
def apply_filters(raw_trajectories, raw_ball_log, launch_time_pct, enable_stats, stats_sens, min_ang, max_ang, min_vel, min_ht, max_ht):
    """Apply all filters to raw data and return filtered results."""

    if not raw_trajectories:
        return [], [], None, None, {}

    filtered_trajs = list(raw_trajectories)
    filtered_log = list(raw_ball_log)
    filter_stats = {
        'initial': len(raw_trajectories),
        'spatial_removed': 0,
        'domain_removed': 0,
        'stats_removed': 0,
        'spatial_details': [],
        'domain_details': [],
        'stats_details': [],
        'launch_zone_info': {},
        'iqr_info': {}
    }

    launch_center = None
    launch_radius = None

    # Step 1: Spatial filtering (launch zone)
    if len(filtered_trajs) >= 3:
        start_positions = np.array([track['path'][0] for track in filtered_trajs])
        start_times = np.array([track['start_time'] for track in filtered_trajs])

        time_threshold = np.percentile(start_times, launch_time_pct)
        early_mask = start_times <= time_threshold
        num_early = np.sum(early_mask)

        if num_early >= 2:
            early_positions = start_positions[early_mask]
            launch_center = np.median(early_positions, axis=0)
            distances_from_center = np.sqrt(np.sum((early_positions - launch_center)**2, axis=1))
            base_radius = np.median(distances_from_center) if len(distances_from_center) > 1 else distances_from_center[0]
            launch_radius = max(base_radius * SPATIAL_FILTER_MULTIPLIER, MIN_LAUNCH_ZONE_RADIUS)
            launch_radius = min(launch_radius, MAX_LAUNCH_ZONE_RADIUS)
        elif len(start_positions) > 0:
            # Fallback: use median of all positions with fixed radius
            launch_center = np.median(start_positions, axis=0)
            launch_radius = 100
        else:
            # Edge case: no trajectories to filter
            return [], [], None, None, filter_stats

        distances = np.sqrt(np.sum((start_positions - launch_center)**2, axis=1))

        new_trajs = []
        new_log = []
        for i, (track, log) in enumerate(zip(filtered_trajs, filtered_log)):
            if distances[i] <= launch_radius:
                new_trajs.append(track)
                new_log.append(log)

        filter_stats['spatial_removed'] = len(filtered_trajs) - len(new_trajs)
        filter_stats['launch_zone_info'] = {
            'center': launch_center,
            'radius': launch_radius,
            'time_pct': launch_time_pct,
            'num_early': num_early
        }
        filtered_trajs = new_trajs
        filtered_log = new_log

    # Step 2: Domain filtering (physically impossible values)
    if len(filtered_trajs) >= 1:
        new_trajs = []
        new_log = []
        domain_details = []

        for track, log in zip(filtered_trajs, filtered_log):
            reasons = []
            angle = log['Launch Angle (°)']
            velocity = log['Initial Velocity (px/s)']
            height = log['Max Height (px from top)']

            # Check if trajectory stayed entirely within launch zone (never left)
            if launch_center is not None and launch_radius is not None:
                path_points = np.array(track['path'])
                distances_from_launch = np.sqrt(np.sum((path_points - launch_center)**2, axis=1))
                max_distance = np.max(distances_from_launch)
                if max_distance <= launch_radius * 1.2:  # Allow 20% buffer
                    reasons.append(f"Never left launch zone (max dist: {max_distance:.0f}px vs zone: {launch_radius:.0f}px)")

            if angle < min_ang or angle > max_ang:
                reasons.append(f"Angle: {angle:.1f}° (range: {min_ang}-{max_ang}°)")
            if velocity < min_vel:
                reasons.append(f"Velocity: {velocity:.1f} px/s (min: {min_vel})")
            if height < min_ht or height > max_ht:
                reasons.append(f"Height: {height} px (range: {min_ht}-{max_ht})")
            if len(track['path']) < MIN_TRAJECTORY_LENGTH:
                reasons.append(f"Length: {len(track['path'])} pts (min: {MIN_TRAJECTORY_LENGTH})")

            if len(reasons) == 0:
                new_trajs.append(track)
                new_log.append(log)
            else:
                domain_details.append({
                    'ball_id': log['Ball #'],
                    'reasons': reasons
                })

        filter_stats['domain_removed'] = len(filtered_trajs) - len(new_trajs)
        filter_stats['domain_details'] = domain_details
        filtered_trajs = new_trajs
        filtered_log = new_log

    # Step 3: Statistical filtering (IQR method)
    if enable_stats and len(filtered_log) >= 5:
        df_stats = pd.DataFrame(filtered_log)
        metrics = ['Initial Velocity (px/s)', 'Launch Angle (°)', 'Max Height (px from top)']
        outlier_mask = np.zeros(len(df_stats), dtype=bool)
        outlier_reasons = [[] for _ in range(len(df_stats))]

        for metric in metrics:
            values = df_stats[metric].values
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - stats_sens * iqr
            upper_bound = q3 + stats_sens * iqr
            is_outlier = (values < lower_bound) | (values > upper_bound)
            outlier_mask |= is_outlier

            # Store IQR info for debugging
            filter_stats['iqr_info'][metric] = {
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower': lower_bound,
                'upper': upper_bound,
                'outliers_count': np.sum(is_outlier),
                'min_value': np.min(values),
                'max_value': np.max(values)
            }

            for i, out in enumerate(is_outlier):
                if out:
                    outlier_reasons[i].append(f"{metric}: {values[i]:.1f} ({lower_bound:.1f}-{upper_bound:.1f})")

        valid_indices = ~outlier_mask
        new_trajs = []
        new_log = []
        stats_details = []

        for i, (track, log) in enumerate(zip(filtered_trajs, filtered_log)):
            if valid_indices[i]:
                new_trajs.append(track)
                new_log.append(log)
            else:
                stats_details.append({
                    'ball_id': log['Ball #'],
                    'reasons': outlier_reasons[i]
                })

        filter_stats['stats_removed'] = len(filtered_trajs) - len(new_trajs)
        filter_stats['stats_details'] = stats_details
        filtered_trajs = new_trajs
        filtered_log = new_log

    return filtered_trajs, filtered_log, launch_center, launch_radius, filter_stats

# Apply filtering after video analysis OR when sliders change
if st.session_state.analysis_complete and st.session_state.raw_trajectories:
    # PHASE 2: Apply interactive filters
    # Debug: show filter parameters
    with st.sidebar.expander("🔧 Active Filter Settings", expanded=False):
        st.caption(f"**Launch Zone:** Earliest {launch_time_percentile}%")
        st.caption(f"**Domain Angle:** {min_angle}° to {max_angle}°")
        st.caption(f"**Domain Height:** {min_height} to {max_height} px")
        st.caption(f"**Domain Velocity:** ≥{min_velocity} px/s")
        st.caption(f"**Stats Filtering:** {'Enabled' if enable_stats_filtering else 'Disabled'}")
        if enable_stats_filtering:
            st.caption(f"**Stats Threshold:** {stats_sensitivity}")

    filtered_trajectories, filtered_log, launch_center, launch_radius, filter_stats = apply_filters(
        st.session_state.raw_trajectories,
        st.session_state.raw_ball_log,
        launch_time_percentile,
        enable_stats_filtering,
        stats_sensitivity,
        min_angle,
        max_angle,
        min_velocity,
        min_height,
        max_height
    )

    # Update session state with filtered results
    st.session_state.all_trajectories = filtered_trajectories
    st.session_state.ball_log = filtered_log
    st.session_state.launch_zone_center = launch_center
    st.session_state.launch_zone_radius = launch_radius

    # Display filtering results with detailed breakdown
    total_removed = filter_stats['spatial_removed'] + filter_stats['domain_removed'] + filter_stats['stats_removed']

    # Debug: Show what's happening with stats filtering
    with st.sidebar.expander("🔍 Debug: Stats Filter Status", expanded=False):
        balls_before_stats = len(st.session_state.raw_trajectories) - filter_stats['spatial_removed'] - filter_stats['domain_removed']
        st.caption(f"**Balls before stats filter:** {balls_before_stats}")
        st.caption(f"**Stats filtering enabled:** {enable_stats_filtering}")
        st.caption(f"**Stats threshold:** {stats_sensitivity}")
        st.caption(f"**Balls removed by stats:** {filter_stats['stats_removed']}")

        if balls_before_stats < 5:
            st.warning(f"⚠️ Need 5+ balls for stats filtering (have {balls_before_stats})")

        # Show IQR details if available
        if 'iqr_info' in filter_stats and filter_stats['iqr_info']:
            st.divider()
            st.caption("**IQR Analysis:**")
            for metric, info in filter_stats['iqr_info'].items():
                metric_short = metric.split('(')[0].strip()
                st.caption(f"**{metric_short}:**")
                st.caption(f"  Range: {info['min_value']:.1f} - {info['max_value']:.1f}")
                st.caption(f"  IQR: {info['iqr']:.1f} (Q1={info['q1']:.1f}, Q3={info['q3']:.1f})")
                st.caption(f"  Bounds: {info['lower']:.1f} - {info['upper']:.1f}")
                st.caption(f"  Outliers: {info['outliers_count']}")
                if info['iqr'] < 1:
                    st.warning(f"⚠️ Very small IQR - data is too consistent!")
        else:
            st.caption("(No IQR data - stats filter didn't run)")

    with st.expander(f"📊 Filtering Summary: {filter_stats['initial']} detected → {len(filtered_trajectories)} valid ({total_removed} removed)", expanded=total_removed > 0):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Spatial Filter", f"{filter_stats['spatial_removed']} removed",
                     delta=f"{len(filtered_trajectories) + filter_stats['domain_removed'] + filter_stats['stats_removed']} kept",
                     delta_color="off")
            if filter_stats['launch_zone_info']:
                lz_info = filter_stats['launch_zone_info']
                st.caption(f"Zone: ({int(lz_info['center'][0])}, {int(lz_info['center'][1])}) ±{int(lz_info['radius'])}px")
                st.caption(f"Using earliest {lz_info['time_pct']}% ({lz_info['num_early']} balls)")

        with col_b:
            st.metric("Domain Filter", f"{filter_stats['domain_removed']} removed",
                     delta=f"{len(filtered_trajectories) + filter_stats['stats_removed']} kept",
                     delta_color="off")
            if filter_stats['domain_details']:
                for detail in filter_stats['domain_details'][:3]:  # Show first 3
                    st.caption(f"Ball #{detail['ball_id']}: {', '.join(detail['reasons'])}")
                if len(filter_stats['domain_details']) > 3:
                    st.caption(f"... and {len(filter_stats['domain_details']) - 3} more")

        with col_c:
            st.metric("Statistical Filter", f"{filter_stats['stats_removed']} removed",
                     delta=f"{len(filtered_trajectories)} final",
                     delta_color="off")
            if filter_stats['stats_details']:
                for detail in filter_stats['stats_details'][:3]:  # Show first 3
                    st.caption(f"Ball #{detail['ball_id']}: {'; '.join(detail['reasons'][:2])}")
                if len(filter_stats['stats_details']) > 3:
                    st.caption(f"... and {len(filter_stats['stats_details']) - 3} more")

    # Draw launch zone on video frame
    if launch_center is not None and st.session_state.last_frame is not None:
        frame_with_zone = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR)
        center_x, center_y = int(launch_center[0]), int(launch_center[1])
        radius = int(launch_radius)
        cv2.circle(frame_with_zone, (center_x, center_y), radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame_with_zone, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.putText(frame_with_zone, "Launch Zone", (center_x - 50, center_y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        display_frame = cv2.cvtColor(frame_with_zone, cv2.COLOR_BGR2RGB)
    else:
        display_frame = st.session_state.last_frame

# Show results if analysis is complete
if st.session_state.analysis_complete and st.session_state.raw_trajectories:
    # Display video frame with launch zone
    if st.session_state.last_frame is not None:
        if 'display_frame' in locals():
            video_feed.image(display_frame)
        else:
            video_feed.image(st.session_state.last_frame)

    # Render chart and summary
    if st.session_state.all_trajectories:
        # Get video dimensions from session state
        width = st.session_state.get('video_width', 1920)
        height = st.session_state.get('video_height', 1080)

        # Calculate accuracy data first (if enabled) so we can add columns to table
        accuracy_data = None
        projected_data = None
        if enable_accuracy and len(st.session_state.all_trajectories) >= 3:
            target_y_px = int(height * st.session_state.target_height_pct / 100)
            accuracy_data = calculate_target_accuracy(
                st.session_state.all_trajectories,
                st.session_state.ball_log,
                target_y_px,
                height
            )
            # New: Calculate Projected Accuracy using early data + physics
            projected_data = calculate_projected_accuracy(
                st.session_state.all_trajectories,
                st.session_state.ball_log,
                target_y_px,
                height
            )

        render_trajectory_chart_unified(st.session_state.all_trajectories, [], st.session_state.ball_log, width, height, enable_accuracy, st.session_state.target_height_pct if enable_accuracy else 40)

        df = pd.DataFrame(st.session_state.ball_log)
        df = df.sort_values("Launch Time (s)").reset_index(drop=True)
        df.insert(0, "Ball", range(1, len(df) + 1))
        # Keep Ball # for now (needed for accuracy mapping), will be dropped by render_summary_unified
        render_summary_unified(df, accuracy_data)

        # Target accuracy visualization
        if enable_accuracy and (accuracy_data is not None or projected_data is not None):
            st.divider()
            tab_actual, tab_projected = st.tabs(["📊 Actual Trajectories", "🔮 Projected (Physics Fit)"])

            with tab_actual:
                if accuracy_data:
                    render_accuracy_analysis(
                        accuracy_data,
                        st.session_state.all_trajectories,
                        st.session_state.ball_log,
                        st.session_state.target_height_pct,
                        width,
                        height,
                        mode="Actual"
                    )
                else:
                    st.info("No trajectories reached the target height directly.")

            with tab_projected:
                if projected_data:
                    st.info("💡 **How it works**: This uses early flight data (launch to apex) to fit a parabolic path. It filters out deviations caused by collisions with the target.")
                    render_accuracy_analysis(
                        projected_data,
                        st.session_state.all_trajectories,
                        st.session_state.ball_log,
                        st.session_state.target_height_pct,
                        width,
                        height,
                        mode="Projected"
                    )
                else:
                    st.warning("Could not calculate physics-based projection for these trajectories.")
        # Summary and save buttons
        initial_count = len(st.session_state.raw_trajectories)
        final_count = len(st.session_state.all_trajectories)
        removed_count = initial_count - final_count

        col_status, col_save_csv, col_save_pdf = st.columns([2, 1, 1])
        with col_status:
            if removed_count > 0:
                st.success(f"✅ **{final_count} valid balls** ({removed_count} filtered)")
            else:
                st.success(f"✅ **{final_count} balls** detected")

        # Save CSV/Chart button
        with col_save_csv:
            if st.button("💾 Save Data", use_container_width=True):
                try:
                    video_dir = os.path.dirname(os.path.abspath(st.session_state.video_path))
                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    # Prepare dataframe for saving (with accuracy data if available)
                    save_df = df.copy()
                    if accuracy_data is not None and 'intercept_map' in accuracy_data:
                        target_distances = []
                        target_positions = []
                        extrapolated_flags = []

                        for _, row in save_df.iterrows():
                            ball_id = row.get('Ball #', None)
                            if ball_id and ball_id in accuracy_data['intercept_map']:
                                info = accuracy_data['intercept_map'][ball_id]
                                target_distances.append(round(info['target_distance'], 1))
                                target_positions.append(round(info['target_x'], 1))
                                extrapolated_flags.append('Yes' if info['extrapolated'] else 'No')
                            else:
                                target_distances.append(None)
                                target_positions.append(None)
                                extrapolated_flags.append('N/A')

                        save_df['Target X (px)'] = target_positions
                        save_df['Target Distance (px)'] = target_distances
                        save_df['Extrapolated'] = extrapolated_flags

                    # Drop Ball # column (internal ID)
                    if 'Ball #' in save_df.columns:
                        save_df = save_df.drop(columns=['Ball #'])

                    # Create filenames with original video name and filter settings for clarity
                    video_name = st.session_state.original_filename
                    filter_desc = f"lz{launch_time_percentile}_angle{min_angle}to{max_angle}_vel{min_velocity}"
                    csv_path = os.path.join(video_dir, f"ball_analysis_{video_name}_{filter_desc}_{timestamp}.csv")
                    save_df.to_csv(csv_path, index=False)

                    # Save trajectory chart (the current filtered view)
                    chart_path = os.path.join(video_dir, f"trajectory_chart_{video_name}_{filter_desc}_{timestamp}.jpg")
                    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')

                    st.session_state.files_saved = True
                    st.session_state.saved_csv = csv_path
                    st.session_state.saved_chart = chart_path
                    st.success(f"✅ Saved!\n\n📊 Chart: `{chart_path}`\n\n📄 Data: `{csv_path}`")
                except Exception as e:
                    st.error(f"❌ Error saving files: {str(e)}")
                st.rerun()

        # Save PDF Report button
        with col_save_pdf:
            if st.button("📄 Save PDF", use_container_width=True):
                try:
                    video_dir = os.path.dirname(os.path.abspath(st.session_state.video_path))
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_name = st.session_state.original_filename
                    pdf_path = os.path.join(video_dir, f"ball_report_{video_name}_{timestamp}.pdf")

                    # Ensure variables are defined (use None/defaults if not available)
                    pdf_accuracy_data = accuracy_data if 'accuracy_data' in locals() else None
                    pdf_filter_stats = filter_stats if 'filter_stats' in locals() else {}
                    pdf_target_height = st.session_state.target_height_pct

                    generate_pdf_report(
                        st.session_state.all_trajectories,
                        st.session_state.ball_log,
                        pdf_accuracy_data,
                        pdf_filter_stats,
                        width,
                        height,
                        pdf_target_height,
                        pdf_path,
                        st.session_state.detection_snapshots
                    )

                    st.session_state.saved_pdf = pdf_path
                    st.success(f"✅ PDF Saved!\n\n📄 Report: `{pdf_path}`")
                except Exception as e:
                    st.error(f"❌ Error saving PDF: {str(e)}")
                st.rerun()

        # Show last saved files info
        if st.session_state.files_saved:
            st.info(f"Last saved:\n\n📊 Chart: `{st.session_state.saved_chart}`\n\n📄 Data: `{st.session_state.saved_csv}`")
        if 'saved_pdf' in st.session_state and st.session_state.saved_pdf:
            st.info(f"📄 PDF Report: `{st.session_state.saved_pdf}`")

        # DO NOT close the global fig - it's reused across reruns
        # plt.close(fig)  # REMOVED - would break subsequent reruns
    else:
        st.warning("No valid trajectories after filtering. Try adjusting filter settings.")
