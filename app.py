import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import pandas as pd

st.set_page_config(layout="wide")
st.title("🏹 Advanced Ball Analytics Dashboard")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Detection")

sensitivity = st.sidebar.slider("Shape Sensitivity", 0.1, 1.0, 0.6, 0.05)
st.sidebar.caption("How circular a blob must be to count as a ball. Increase if non-ball objects are being detected; decrease if real balls are being missed.")

sat_val = st.sidebar.slider("Yellow Threshold", 50, 255, 120)
st.sidebar.caption("Minimum color saturation to detect as yellow. Increase in bright/sunny conditions to reduce false positives; decrease if balls are being missed in shade.")

st.sidebar.header("Tracking")

match_threshold = st.sidebar.slider("Match Distance (px)", 10, 300, 80)
st.sidebar.caption("How far a ball can move between frames and still be considered the same ball. Decrease if separate balls are merging into one track; increase if fast balls are losing their track.")

launch_zone_height = st.sidebar.slider("Launch Zone Height (px)", 20, 400, 150)
st.sidebar.caption("Vertical size of the launch zone box. The box snaps to the right edge of the frame at the first detected ball. Only balls appearing inside it start a new track.")

memory_frames = st.sidebar.slider("Gap Tolerance (frames)", 1, 30, 15)
st.sidebar.caption("How many frames a ball can disappear (e.g. behind an object) before its track is finalized. Increase if tracks are ending prematurely.")

st.sidebar.header("Playback")

speed = st.sidebar.slider("Playback Speed", 0.1, 5.0, 1.0)
st.sidebar.caption("Speed multiplier for video playback. Does not affect analysis accuracy.")

auto_fps = st.sidebar.checkbox("Auto-detect FPS", value=True)
st.sidebar.caption("Read FPS from the video file. Uncheck only if velocity/timing values look wrong.")
manual_fps = st.sidebar.number_input("Manual FPS Override", value=30) if not auto_fps else 30

# Initialize session state for persistent data
if 'all_trajectories' not in st.session_state:
    st.session_state.all_trajectories = []
if 'ball_log' not in st.session_state:
    st.session_state.ball_log = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
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
if 'launch_zone' not in st.session_state:
    st.session_state.launch_zone = None        # (center_x, center_y) once established
if 'launch_zone_origins' not in st.session_state:
    st.session_state.launch_zone_origins = []  # starting positions of first 2 finalized balls

if st.sidebar.button("Reset All Data"):
    st.session_state.all_trajectories = []
    st.session_state.ball_log = []
    st.session_state.active_tracks = []
    st.session_state.pause_frame = 0
    st.session_state.paused = False
    st.session_state.next_ball_id = 1
    st.session_state.last_frame = None
    st.session_state.launch_zone = None
    st.session_state.launch_zone_origins = []
    st.rerun()


uploaded_file = st.file_uploader("Upload your MOV file", type=['mov', 'mp4'])

if uploaded_file:
    temp_path = "temp_video.mov"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.video_path = temp_path
elif st.session_state.video_path:
    temp_path = st.session_state.video_path
else:
    temp_path = None

if temp_path:
    col_run, col_rerun = st.columns([3, 1])
    with col_rerun:
        if st.button("Rerun Analysis", disabled=not st.session_state.video_path):
            st.session_state.all_trajectories = []
            st.session_state.ball_log = []
            st.session_state.active_tracks = []
            st.session_state.pause_frame = 0
            st.session_state.paused = False
            st.session_state.next_ball_id = 1
            st.session_state.last_frame = None
            st.session_state.launch_zone = None

    cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        st.error("Failed to open video file.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) if auto_fps else manual_fps
        if video_fps <= 0: video_fps = 30
        
        # UI Layout

        col1, col2 = st.columns([1, 1])
        video_feed = col1.empty()
        graph_plot = col2.empty()

        pause_btn = st.empty()
        status_bar = st.empty()
        summary_area = st.empty()
        trend_plot = st.empty()
        log_table = st.empty()

        lower_yellow = np.array([20, sat_val, 100]) 
        upper_yellow = np.array([35, 255, 255])
        
        active_tracks = st.session_state.active_tracks if st.session_state.pause_frame > 0 else []
        fig, ax = plt.subplots(figsize=(6, 5))
        colormap = cm.get_cmap('gist_rainbow')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        GRAPH_UPDATE_INTERVAL = 10

        # Seek to paused position if resuming
        frame_count = st.session_state.pause_frame
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if st.session_state.paused:
            if st.session_state.last_frame is not None:
                video_feed.image(st.session_state.last_frame)
            if st.session_state.all_trajectories or st.session_state.active_tracks:
                ax.clear()
                ax.set_xlim(0, width)
                ax.set_ylim(-height, 0)
                ax.set_facecolor('#1e1e1e')
                for track in st.session_state.all_trajectories:
                    pts = np.array(track['path'])
                    ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=2, alpha=0.85)
                    mid = len(pts) // 2
                    ax.text(pts[mid, 0], -pts[mid, 1], str(track['id']),
                            color=track['color'], fontsize=7, ha='center', va='bottom')
                for track in st.session_state.active_tracks:
                    if len(track['path']) > 1:
                        pts = np.array(track['path'])
                        ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=4)
                        mid = len(pts) // 2
                        ax.text(pts[mid, 0], -pts[mid, 1], str(track['id']),
                                color=track['color'], fontsize=7, ha='center', va='bottom')
                graph_plot.pyplot(fig)
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
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_centers = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if area > 100 and perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > sensitivity:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            current_centers.append(center)

            # 2. Tracking & Velocity Calculation
            max_track_frames = int(video_fps * 3)  # max 3s per ball flight

            def predict_pos(path):
                if len(path) >= 2:
                    dx = path[-1][0] - path[-2][0]
                    dy = path[-1][1] - path[-2][1]
                    return (path[-1][0] + dx, path[-1][1] + dy)
                return path[-1]

            def direction_ok(path, new_pos, max_angle=120):
                if len(path) < 2:
                    return True
                vx = path[-1][0] - path[-2][0]
                vy = path[-1][1] - path[-2][1]
                dx = new_pos[0] - path[-1][0]
                dy = new_pos[1] - path[-1][1]
                if (vx == 0 and vy == 0) or (dx == 0 and dy == 0):
                    return True
                cos_a = (vx*dx + vy*dy) / ((vx**2+vy**2)**0.5 * (dx**2+dy**2)**0.5)
                return np.degrees(np.arccos(np.clip(cos_a, -1, 1))) < max_angle

            def has_bounced(path, consistent_frames=3, min_speed=2):
                """Detect any surface contact by spotting a velocity reversal in x or y.

                - Ground bounce: ball was falling (dy > 0) then rises (dy < 0)
                - Wall bounce:   ball was moving in one x direction then reverses
                - Natural arc peak (dy negative -> positive) is intentionally ignored
                  because that is normal ballistic flight, not a contact event.
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

                ground_bounce = reversed_after_consistent(dys, was_positive=True)   # falling -> rising
                wall_bounce   = (reversed_after_consistent(dxs, was_positive=True)  # right -> left
                              or reversed_after_consistent(dxs, was_positive=False)) # left -> right
                return ground_bounce or wall_bounce

            def finalize(track):
                p = track['path']
                # Initial velocity: average speed over first 3 frames for stability
                n = min(4, len(p))
                init_vel = sum(
                    np.linalg.norm(np.array(p[i+1]) - np.array(p[i])) * video_fps
                    for i in range(n - 1)
                ) / (n - 1) if n > 1 else 0
                # Launch angle: direction of initial movement (averaged over first 3 frames)
                # Positive = upward (y decreases going up in image coords)
                if len(p) >= 2:
                    dx = np.mean([p[i+1][0] - p[i][0] for i in range(n - 1)])
                    dy = np.mean([p[i+1][1] - p[i][1] for i in range(n - 1)])
                    launch_angle = round(np.degrees(np.arctan2(-dy, dx)), 1)  # negate dy: up = positive
                else:
                    launch_angle = 0.0
                # Max height: minimum y in path (y=0 is top of frame)
                max_height_px = min(pt[1] for pt in p)
                st.session_state.all_trajectories.append(track)
                st.session_state.ball_log.append({
                    "Ball #": track['id'],
                    "Launch Time (s)": round(track['start_time'], 2),
                    "Initial Velocity (px/s)": round(init_vel, 1),
                    "Launch Angle (°)": launch_angle,
                    "Max Height (px from top)": max_height_px,
                })
                # After the 2nd ball finalizes, lock the launch zone from their origins
                if st.session_state.launch_zone is None and len(st.session_state.launch_zone_origins) < 2:
                    st.session_state.launch_zone_origins.append(track['path'][0])
                    if len(st.session_state.launch_zone_origins) == 2:
                        origins = st.session_state.launch_zone_origins
                        # Only calibrate vertical center; horizontal left edge stays
                        # pinned to the same right-third used during calibration so
                        # later balls aren't rejected for appearing slightly further left
                        cy = int(np.mean([o[1] for o in origins]))
                        st.session_state.launch_zone = cy

            new_active = []
            for center in current_centers:
                matched = False
                best_dist = match_threshold
                best_track = None
                for track in active_tracks:
                    predicted = predict_pos(track['path'])
                    dist = np.linalg.norm(np.array(center) - np.array(predicted))
                    if dist < best_dist and direction_ok(track['path'], center):
                        best_dist = dist
                        best_track = track
                if best_track is not None:
                    best_track['path'].append(center)
                    best_track['missing_count'] = 0
                    new_active.append(best_track)
                    active_tracks.remove(best_track)
                    matched = True

                if not matched:
                    if st.session_state.launch_zone is None:
                        # Calibration phase: only accept detections in the right third of the
                        # frame so bounced/stray balls in the middle don't corrupt calibration
                        in_zone = center[0] >= width * 2 // 3
                    else:
                        zy = st.session_state.launch_zone
                        half_h = launch_zone_height
                        in_zone = (center[0] >= width * 2 // 3 and
                                   zy - half_h <= center[1] <= zy + half_h)
                    if in_zone:
                        ball_id = st.session_state.next_ball_id
                        st.session_state.next_ball_id += 1
                        new_active.append({
                            'id': ball_id,
                            'path': [center],
                            'color': colormap((ball_id * 0.618033988749895) % 1.0),
                            'missing_count': 0,
                            'start_time': current_timestamp
                        })

            for track in active_tracks:
                track['missing_count'] += 1
                track_age = len(track['path']) + track['missing_count']
                expired = track_age > max_track_frames
                lost = track['missing_count'] >= memory_frames
                if (expired or lost) and len(track['path']) > 4:
                    finalize(track)
                elif not expired and not lost:
                    new_active.append(track)
            
            # Finalize tracks whose ballistic arc was interrupted by a bounce
            still_flying = []
            for track in new_active:
                if has_bounced(track['path']) and len(track['path']) > 4:
                    finalize(track)
                else:
                    still_flying.append(track)
            new_active = still_flying

            active_tracks = new_active
            st.session_state.active_tracks = active_tracks
            st.session_state.pause_frame = frame_count

            # Draw launch zone box (shown only once it's locked from 2 calibration balls)
            if st.session_state.launch_zone is not None:
                zy = st.session_state.launch_zone
                half_h = launch_zone_height
                bx1, by1, bx2, by2 = width * 2 // 3, zy - half_h, width, zy + half_h
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "launch zone", (max(bx1 - 4, 0), by1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            else:
                remaining = 2 - len(st.session_state.launch_zone_origins)
                cv2.putText(frame, f"calibrating: {remaining} ball(s) to go",
                            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Draw tracking boxes with ball numbers
            for track in active_tracks:
                center = track['path'][-1]
                r, g, b, _ = track['color']
                color_bgr = (int(b * 255), int(g * 255), int(r * 255))
                cv2.circle(frame, center, 15, color_bgr, 2)
                label = str(track['id'])
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, label, (center[0] - tw // 2, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            # 3. UI Updates
            total_ball_count = len(st.session_state.all_trajectories) + len(active_tracks)
            cv2.putText(frame, f"Time: {current_timestamp:.2f}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"Balls: {total_ball_count}", (50, 100),
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
                ax.clear()
                ax.set_xlim(0, width)
                ax.set_ylim(-height, 0)
                ax.set_facecolor('#1e1e1e')

                for track in st.session_state.all_trajectories:
                    pts = np.array(track['path'])
                    ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=2, alpha=0.85)
                    mid = len(pts) // 2
                    ax.text(pts[mid, 0], -pts[mid, 1], str(track['id']),
                            color=track['color'], fontsize=7, ha='center', va='bottom')

                for track in active_tracks:
                    if len(track['path']) > 1:
                        pts = np.array(track['path'])
                        ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=4)
                        mid = len(pts) // 2
                        ax.text(pts[mid, 0], -pts[mid, 1], str(track['id']),
                                color=track['color'], fontsize=7, ha='center', va='bottom')

                graph_plot.pyplot(fig)

            t_graph_end = time.time()

            # Update Log Table + Summary
            if st.session_state.ball_log and frame_count % GRAPH_UPDATE_INTERVAL == 0:
                df = pd.DataFrame(st.session_state.ball_log)
                df = df.sort_values("Launch Time (s)").reset_index(drop=True)
                df.insert(0, "Ball", range(1, len(df) + 1))
                df = df.drop(columns=["Ball #"])

                with summary_area.container():
                    st.markdown("**Summary**")
                    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                    c1.metric("Balls", len(df))
                    c2.metric("Avg Velocity", f"{df['Initial Velocity (px/s)'].mean():.0f} px/s")
                    c3.metric("Velocity Std Dev", f"{df['Initial Velocity (px/s)'].std():.0f} px/s")
                    c4.metric("Avg Launch Angle", f"{df['Launch Angle (°)'].mean():.1f}°")
                    c5.metric("Angle Std Dev", f"{df['Launch Angle (°)'].std():.1f}°")
                    c6.metric("Best Height", f"{df['Max Height (px from top)'].min()} px")
                    c7.metric("Height Std Dev", f"{df['Max Height (px from top)'].std():.0f} px")

                if len(df) >= 2:
                    fig_trend, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
                    fig_trend.patch.set_facecolor('#1e1e1e')
                    x = df["Ball"]
                    for ax in (ax1, ax2, ax3):
                        ax.set_facecolor('#1e1e1e')
                        ax.tick_params(colors='white')
                        ax.spines[:].set_color('#444')

                    ax1.plot(x, df["Initial Velocity (px/s)"], 'o-', color='#00bfff')
                    ax1.axhline(df["Initial Velocity (px/s)"].mean(), color='#00bfff', linestyle='--', alpha=0.4)
                    ax1.set_ylabel("Init Velocity\n(px/s)", color='white', fontsize=8)

                    ax2.plot(x, df["Launch Angle (°)"], 'o-', color='#ff7f0e')
                    ax2.axhline(df["Launch Angle (°)"].mean(), color='#ff7f0e', linestyle='--', alpha=0.4)
                    ax2.set_ylabel("Launch Angle\n(°)", color='white', fontsize=8)

                    ax3.plot(x, df["Max Height (px from top)"], 'o-', color='#2ecc71')
                    ax3.axhline(df["Max Height (px from top)"].mean(), color='#2ecc71', linestyle='--', alpha=0.4)
                    ax3.set_ylabel("Max Height\n(px from top)", color='white', fontsize=8)
                    ax3.set_xlabel("Ball #", color='white', fontsize=9)
                    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    fig_trend.tight_layout()
                    trend_plot.pyplot(fig_trend)
                    plt.close(fig_trend)

                log_table.dataframe(df, hide_index=True, use_container_width=True)

            t_total = time.time() - t_frame_start
            print(f"[frame {frame_count:04d}] total={t_total*1000:.1f}ms | cv={( t_cv - t_frame_start)*1000:.1f}ms | video_feed={(t_video - t_cv)*1000:.1f}ms | graph={(t_graph_end - t_graph_start)*1000:.1f}ms | balls={len(st.session_state.all_trajectories)}")

            # Sync Playback
            elapsed = time.time() - start_process_time
            delay = max(0, (1.0 / (video_fps * speed)) - elapsed)
            time.sleep(delay)

        cap.release()

        # Finalize any tracks still in flight when the video ended
        for track in active_tracks:
            if len(track['path']) > 4:
                finalize(track)
        active_tracks = []
        st.session_state.active_tracks = []

        # Final chart render with all trajectories
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(-height, 0)
        ax.set_facecolor('#1e1e1e')
        for track in st.session_state.all_trajectories:
            pts = np.array(track['path'])
            ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=2, alpha=0.85)
            mid = len(pts) // 2
            ax.text(pts[mid, 0], -pts[mid, 1], str(track['id']),
                    color=track['color'], fontsize=7, ha='center', va='bottom')
        graph_plot.pyplot(fig)

        status_bar.empty()
        st.success(f"Analysis complete — {len(st.session_state.all_trajectories)} balls detected.")