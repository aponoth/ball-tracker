import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_yellow_balls(video_path):
    cap = cv2.VideoCapture(video_path)
    # Define Yellow in HSV color space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    trajectories = [] # Stores list of (x, y) for each ball
    active_tracks = [] # Currently moving balls: {'path': [], 'last_pos': (x,y)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Isolate Yellow
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_balls = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50: # Ignore noise
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    current_balls.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        # 2. Simple Tracker: Match current balls to existing paths
        new_active_tracks = []
        for ball in current_balls:
            matched = False
            for track in active_tracks:
                # If ball is close to where a ball was in the last frame
                dist = np.linalg.norm(np.array(ball) - np.array(track['last_pos']))
                if dist < 100: 
                    track['path'].append(ball)
                    track['last_pos'] = ball
                    new_active_tracks.append(track)
                    active_tracks.remove(track)
                    matched = True
                    break
            if not matched:
                new_active_tracks.append({'path': [ball], 'last_pos': ball})
        
        # Save completed trajectories
        for track in active_tracks:
            if len(track['path']) > 5: # Only keep significant paths
                trajectories.append(track['path'])
        active_tracks = new_active_tracks

    cap.release()
    return trajectories

# Visualization
def plot_trajectories(trajectories):
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(trajectories):
        path = np.array(path)
        plt.plot(path[:, 0], -path[:, 1], label=f"Ball {i+1}") # Negative Y to flip image coords
    
    plt.title("Ball Launch Trajectories")
    plt.xlabel("Horizontal Position (Pixels)")
    plt.ylabel("Vertical Position (Pixels)")
    plt.legend()
    plt.show()

import streamlit as st

st.title("⚽ Ball Trajectory Tracker")
uploaded_file = st.file_uploader("Upload your MOV file", type=['mov', 'mp4'])

if uploaded_file:
    with open("temp_video.mov", "wb") as f:
        f.write(uploaded_file.read())
    
    st.write("Processing trajectories...")
    data = track_yellow_balls("temp_video.mov")
    
    # Create the graph
    fig, ax = plt.subplots()
    for path in data:
        pts = np.array(path)
        ax.plot(pts[:, 0], -pts[:, 1])
    
    st.pyplot(fig)