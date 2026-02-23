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
st.sidebar.header("Tuning Controls")
sensitivity = st.sidebar.slider("Detection Sensitivity (Circularity)", 0.1, 1.0, 0.6, 0.05)
sat_val = st.sidebar.slider("Color Saturation Floor", 50, 255, 120)
memory_frames = st.sidebar.slider("Memory (Frames to bridge gaps)", 1, 30, 15)

st.sidebar.header("Playback & Timing")
speed = st.sidebar.slider("Playback Speed", 0.1, 5.0, 1.0)
auto_fps = st.sidebar.checkbox("Auto-detect FPS from Video", value=True)
manual_fps = st.sidebar.number_input("Manual FPS Override", value=30) if not auto_fps else 30

# Initialize session state for persistent data
if 'all_trajectories' not in st.session_state:
    st.session_state.all_trajectories = []
if 'ball_log' not in st.session_state:
    st.session_state.ball_log = []

if st.sidebar.button("Reset All Data"):
    st.session_state.all_trajectories = []
    st.session_state.ball_log = []
    st.rerun()

uploaded_file = st.file_uploader("Upload your MOV file", type=['mov', 'mp4'])

if uploaded_file:
    temp_path = "temp_video.mov"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        st.error("Failed to open video file.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) if auto_fps else manual_fps
        if video_fps <= 0: video_fps = 30
        
        # UI Layout
        m1, m2 = st.columns(2)
        m1.metric("Total Balls Detected", len(st.session_state.all_trajectories))
        
        col1, col2 = st.columns([1, 1])
        video_feed = col1.empty()
        graph_plot = col2.empty()
        
        log_table = st.empty() # Placeholder for the data table

        lower_yellow = np.array([20, sat_val, 100]) 
        upper_yellow = np.array([35, 255, 255])
        
        active_tracks = []
        fig, ax = plt.subplots(figsize=(6, 5))
        colormap = cm.get_cmap('gist_rainbow')
        frame_count = 0

        while cap.isOpened():
            start_process_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            current_timestamp = frame_count / video_fps

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
                            cv2.circle(frame, center, 15, (0, 255, 0), 2)

            # 2. Tracking & Velocity Calculation
            new_active = []
            for center in current_centers:
                matched = False
                for track in active_tracks:
                    last_pos = track['path'][-1]
                    dist = np.linalg.norm(np.array(center) - np.array(last_pos))
                    if dist < 200:
                        track['path'].append(center)
                        track['missing_count'] = 0
                        new_active.append(track)
                        active_tracks.remove(track)
                        matched = True
                        break
                
                if not matched:
                    ball_id = len(st.session_state.all_trajectories) + len(new_active) + 1
                    new_active.append({
                        'id': ball_id,
                        'path': [center], 
                        'color': colormap((ball_id * 0.15) % 1.0),
                        'missing_count': 0,
                        'start_time': current_timestamp
                    })

            for track in active_tracks:
                track['missing_count'] += 1
                if track['missing_count'] < memory_frames:
                    new_active.append(track)
                elif len(track['path']) > 8:
                    # Finalize Ball Stats
                    total_dist = 0
                    p = track['path']
                    for i in range(len(p)-1):
                        total_dist += np.linalg.norm(np.array(p[i+1]) - np.array(p[i]))
                    
                    duration = len(p) / video_fps
                    avg_velocity = total_dist / duration if duration > 0 else 0
                    
                    st.session_state.all_trajectories.append(track)
                    st.session_state.ball_log.append({
                        "Ball #": track['id'],
                        "Launch Time (s)": round(track['start_time'], 2),
                        "Avg Velocity (px/s)": round(avg_velocity, 1)
                    })
            
            active_tracks = new_active

            # 3. UI Updates
            cv2.putText(frame, f"Time: {current_timestamp:.2f}s", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, f"Balls: {len(st.session_state.all_trajectories)}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            ax.clear()
            ax.set_xlim(0, width)
            ax.set_ylim(-height, 0)
            ax.set_facecolor('#1e1e1e') # Dark mode graph
            
            for track in st.session_state.all_trajectories:
                pts = np.array(track['path'])
                ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=1, alpha=0.4)
            
            for track in active_tracks:
                if len(track['path']) > 1:
                    pts = np.array(track['path'])
                    ax.plot(pts[:, 0], -pts[:, 1], color=track['color'], linewidth=4)

            graph_plot.pyplot(fig)
            
            # Update Log Table
            if st.session_state.ball_log:
                log_table.table(pd.DataFrame(st.session_state.ball_log).tail(10))

            # Sync Playback
            elapsed = time.time() - start_process_time
            delay = max(0, (1.0 / (video_fps * speed)) - elapsed)
            time.sleep(delay)

        cap.release()
        st.success(f"Final Count: {len(st.session_state.all_trajectories)} balls detected.")