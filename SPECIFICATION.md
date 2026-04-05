# Ball Tracker - Technical Specification

**Version**: 1.0  
**Last Updated**: 2026-04-05  
**Status**: Production

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Features](#3-core-features)
4. [Data Models](#4-data-models)
5. [Algorithms](#5-algorithms)
6. [User Interface](#6-user-interface)
7. [Configuration Parameters](#7-configuration-parameters)
8. [Performance Requirements](#8-performance-requirements)
9. [Export Formats](#9-export-formats)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Overview

### 1.1 Purpose
Ball Tracker is a video analysis application for tracking and analyzing ball trajectories with sub-pixel precision. It uses computer vision to detect yellow balls in video, tracks their paths, and provides detailed analytics on launch characteristics and landing accuracy.

### 1.2 Primary Use Cases
- Analyze shooter performance metrics (velocity, angle, height, accuracy)
- Track projectile motion in sports/gaming scenarios (cornhole, archery, etc.)
- Filter data for quality using spatial, domain, and statistical filters
- Generate physics-based trajectory projections accounting for gravity and Magnus effect

### 1.3 Technology Stack
- **Language**: Python 3.6+
- **UI Framework**: Streamlit 1.41.1
- **Computer Vision**: OpenCV 4.10.0.84
- **Numerical Computing**: NumPy 2.0.1, Pandas 2.2.3
- **Visualization**: Matplotlib 3.9.2, Plotly (graph_objects)
- **PDF Generation**: Matplotlib PdfPages

---

## 2. System Architecture

### 2.1 Two-Phase Architecture

```
┌─────────────────────────────────────────┐
│  PHASE 1: Video Analysis (Live)        │
│  - Real-time ball detection             │
│  - Multi-target tracking                │
│  - Trajectory finalization              │
│  - Stores RAW immutable data            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  PHASE 2: Interactive Filtering         │
│  - Spatial filtering (launch zone)      │
│  - Domain filtering (physics)           │
│  - Statistical filtering (IQR)          │
│  - Real-time filter updates             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Analysis & Visualization               │
│  - Trajectory charts                    │
│  - Summary statistics                   │
│  - Target accuracy analysis             │
│  - Physics-based projections            │
│  - Correlation analysis                 │
└─────────────────────────────────────────┘
```

### 2.2 Data Flow

**Phase 1 (Detection) → RAW Data**:
```
Video → Frame → HSV Color Space → Threshold → Contours → 
Centers → Track Matching → Velocity Prediction → Bounce Detection → 
Finalize → RAW Trajectories + RAW Ball Log
```

**Phase 2 (Filtering) → Filtered Data**:
```
RAW Data → Spatial Filter → Domain Filter → Statistical Filter → 
Filtered Trajectories + Filtered Ball Log
```

### 2.3 Session State Management

**Immutable RAW Data**:
- `raw_trajectories`: Never modified after Phase 1
- `raw_ball_log`: Never modified after Phase 1

**Filtered Data** (Rebuilt on slider changes):
- `all_trajectories`: Filtered subset of raw_trajectories
- `ball_log`: Filtered subset of raw_ball_log

**Tracking State**:
- `active_tracks`: Currently tracked balls in flight
- `next_ball_id`: Sequential ID counter
- `detection_snapshots`: Sample frames for PDF

**UI State**:
- `paused`, `pause_frame`, `last_frame`
- `launch_zone_center`, `launch_zone_radius`
- `analysis_complete`, `files_saved`
- `video_view_type`: "Side View" or "Down-the-Line (DTL)"

---

## 3. Core Features

### 3.1 Multi-Ball Tracking

**Algorithm**: Global Greedy Data Association
- Computes all potential center-track pairs
- Sorts by distance (best matches first)
- Assigns one-to-one mapping globally (prevents cascade failures)

**Capabilities**:
- Tracks multiple balls simultaneously
- Handles occlusions (15-frame memory buffer)
- Predicts next position using velocity + gravity
- Validates direction continuity (max 120° angle change)
- Detects bounces via velocity reversal

**Ball Identification**:
- Sequential ID assignment on first detection
- Color-coded using golden ratio (0.618...) for visual distinction
- Chronological renumbering by launch time for display

### 3.2 Color Detection

**Target**: Yellow balls
**Method**: HSV color space thresholding

**Parameters**:
- Hue range: [20, 35]
- Saturation range: [sat_val, 255] (user adjustable: 50-255)
- Value range: [100, 255]

**Processing Pipeline**:
1. BGR → HSV conversion
2. Color threshold mask
3. Morphological opening (5×5 kernel) to remove noise
4. Contour detection
5. Circularity filter (shape_sensitivity: 0.1-1.0)

### 3.3 Trajectory Analysis

**Metrics Computed**:
- **Initial Velocity**: Average over first 4 points × FPS (px/s)
- **Launch Angle**: atan2(-dy, dx) in degrees
- **Max Height**: Minimum Y coordinate (px from top)
- **Flight Time**: Duration from launch to finalize (seconds)
- **Launch Time**: Timestamp when ball first detected (seconds)

**Special Handling for DTL (Down-the-Line) View**:
- Launch angle uses abs(dx) for steepness regardless of side
- Match threshold increased 6× (handles 500+ pixel jumps)
- Min ball area reduced to 0.3× (distant balls appear smaller)
- Circularity sensitivity reduced to 0.4× (streaking balls)
- Memory frames increased 3× (handle flickering)
- Direction validation disabled (trust position matching)
- Velocity decay applied (0.98× per frame for depth effects)

### 3.4 Filtering System

#### 3.4.1 Spatial Filter (Launch Zone)

**Purpose**: Remove trajectories that don't originate from the launch area

**Algorithm**:
1. Sort trajectories by launch time
2. Select earliest X% (user configurable: 10-50%)
3. Calculate median center position of early trajectories
4. Calculate median distance from center
5. Set radius = median_distance × 3.0 (clamped: 50-200px)
6. Remove trajectories starting outside radius

**Output**: Launch zone center (x, y) and radius (px)

#### 3.4.2 Domain Filter (Physics-Based)

**Purpose**: Remove physically impossible or invalid trajectories

**Filters**:
1. **Angle Range**: User-defined min/max launch angle (default: 20-80°)
2. **Height Range**: User-defined min/max height (default: 0-1200px)
3. **Velocity Threshold**: Minimum initial velocity (default: 10 px/s)
4. **Launch Zone Exit**: Trajectory must leave launch zone (120% radius buffer)
5. **Minimum Length**: At least 5 points (MIN_TRAJECTORY_LENGTH)
6. **Displacement**: Total movement ≥ 50px × (30/fps) - filters stationary detections

#### 3.4.3 Statistical Filter (IQR Outlier Removal)

**Purpose**: Remove statistical outliers across metrics

**Method**: Interquartile Range (IQR)
- Enabled/disabled by user (checkbox)
- Threshold: 1.5-4.0 (user adjustable, default: 2.5)
- Requires ≥5 trajectories

**Algorithm**:
```
For each metric (Velocity, Angle, Height):
  Q1 = 25th percentile
  Q3 = 75th percentile
  IQR = Q3 - Q1
  Lower_bound = Q1 - threshold × IQR
  Upper_bound = Q3 + threshold × IQR
  Remove if value < Lower_bound OR value > Upper_bound
```

**Metrics Analyzed**:
- Initial Velocity (px/s)
- Launch Angle (°)
- Max Height (px from top)

### 3.5 Target Accuracy Analysis

#### 3.5.1 Actual Trajectory Intercept

**Purpose**: Calculate where actual trajectories cross target height

**Algorithm**:
1. Find apex (minimum Y value)
2. Extract descending portion (apex → end)
3. Find intercept on descending path (linear interpolation)
4. If trajectory ends early, extrapolate using last 3 points

**Extrapolation** (when trajectory ends before target):
```python
vx = (p3[0] - p1[0]) / 2  # Horizontal velocity
vy = (p3[1] - p1[1]) / 2  # Vertical velocity
steps = (target_y - p3[1]) / vy
x_intercept = p3[0] + vx × steps
```
- Only extrapolate if descending (vy > 0.5)
- Limit: max 50 frames extrapolation

#### 3.5.2 Physics-Based Projection

**Purpose**: Project trajectory using physics model fitted to actual data

**Physics Model**:
```
x(t) = x0 + vx × t                    (constant horizontal velocity)
y(t) = y0 + vy0 × t + 0.5 × g_eff × t²  (parabolic with gravity)
```

**Fitting Algorithm**:
1. **Extract Ascending Phase**: Launch → apex only
2. **Fit Initial Conditions**:
   - x0 = actual starting X position
   - y0 = actual starting Y position
   - vx = slope of (x - x0) vs t (linear fit)
   - Fit quadratic: (y - y0) = a×t² + b×t + c
   - g_eff = 2×a (effective gravity per trajectory)
   - vy0 = b (initial vertical velocity)
   - Validate: |c| < 10 (intercept should be near zero)

3. **Project Descending Trajectory**:
   - Solve quadratic: y0 + vy0×t + 0.5×g_eff×t² = target_y
   - Two roots: t1 (ascending), t2 (descending)
   - Select t2 (after apex: t > -vy0/g_eff)
   - x_projected = x0 + vx × t2

**Effective Gravity** (g_eff):
- Fitted per trajectory (not hardcoded)
- Captures: true gravity + Magnus lift + air resistance
- For backspin balls: g_eff < 0.5 (upward lift reduces effective gravity)
- Ensures perfect overlap from launch → apex

**Metrics Computed**:
- Mean X position
- Standard deviation (spread)
- CEP (50%): Median distance from mean
- R95 (95%): 95th percentile distance from mean
- Min/Max/Spread in X dimension

### 3.6 Correlation Analysis

**Purpose**: Identify which factors influence landing accuracy

**Method**: Pearson correlation between metrics and target X position

**Metrics Analyzed**:
1. Initial Velocity vs Target X
2. Launch Angle vs Target X
3. Max Height vs Target X
4. Launch Time vs Target X

**Visualization**:
- Scatter plots with ball number labels
- Trend lines for |correlation| > 0.1
- Correlation coefficient displayed on each subplot

---

## 4. Data Models

### 4.1 Track Object (During Detection)

```python
track = {
    'id': int,                    # Sequential ball ID
    'path': [(x, y), ...],        # List of (x, y) tuples
    'color': (r, g, b, a),        # RGBA tuple (0-1 range)
    'missing_count': int,         # Frames without detection
    'start_time': float          # Timestamp in seconds
}
```

### 4.2 Ball Log Entry (Analysis Output)

```python
ball_log_entry = {
    'Ball #': int,                           # Internal ID
    'Launch Time (s)': float,                # Timestamp
    'Initial Velocity (px/s)': float,        # Speed
    'Launch Angle (°)': float,               # Angle in degrees
    'Max Height (px from top)': int,         # Apex Y coordinate
    'View Type': str                         # "Side View" or "DTL"
}
```

### 4.3 Accuracy Data Object

```python
accuracy_data = {
    'intercepts': [                          # List of intercept points
        {
            'ball_id': int,
            'ball_num': int,                 # Sequential number
            'x': float,                      # X position at target
            'y': float,                      # Y position (=target_y)
            'launch_time': float,
            'extrapolated': bool,
            'is_projection': bool,           # True for physics-based
            'physics_params': {              # Only for projections
                'x0': float,
                'y0': float,
                'vx': float,
                'vy0': float,
                'g_eff': float,              # Fitted gravity
                'apex_idx': int,
                't_intercept': float
            }
        }
    ],
    'mean_x': float,                         # Average landing position
    'std_x': float,                          # Standard deviation
    'cep': float,                            # 50% error radius
    'r95': float,                            # 95% error radius
    'min_x': float,
    'max_x': float,
    'spread': float,
    'intercept_map': {                       # For table joins
        ball_id: {
            'target_x': float,
            'target_distance': float,
            'extrapolated': bool
        }
    }
}
```

### 4.4 Filter Statistics Object

```python
filter_stats = {
    'initial': int,                          # Count before filtering
    'spatial_removed': int,
    'domain_removed': int,
    'stats_removed': int,
    'spatial_details': [                     # Per-ball reasons
        {'ball_id': int, 'reasons': [str]}
    ],
    'domain_details': [...],
    'stats_details': [...],
    'launch_zone_info': {
        'center': (float, float),
        'radius': float,
        'time_pct': float,
        'num_early': int
    },
    'iqr_info': {                            # Per-metric IQR data
        'metric_name': {
            'q1': float,
            'q3': float,
            'iqr': float,
            'lower': float,
            'upper': float,
            'outliers_count': int,
            'min_value': float,
            'max_value': float
        }
    }
}
```

---

## 5. Algorithms

### 5.1 Ball Detection (Per Frame)

```python
def detect_balls(frame, sat_val, sensitivity, min_area):
    # 1. Color space conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Threshold yellow
    lower = [20, sat_val, 100]
    upper = [35, 255, 255]
    mask = cv2.inRange(hsv, lower, upper)
    
    # 3. Morphological opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5x5)
    
    # 4. Find contours
    contours = cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    # 5. Filter by circularity
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        
        if area > min_area and perimeter > 0:
            circularity = 4π × area / perimeter²
            
            if circularity > sensitivity:
                moments = cv2.moments(contour)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centers.append((cx, cy))
    
    return centers
```

### 5.2 Global Greedy Data Association

```python
def associate_detections(centers, active_tracks, threshold):
    # 1. Build all potential matches
    matches = []
    for c_idx, center in enumerate(centers):
        for t_idx, track in enumerate(active_tracks):
            predicted = predict_position(track)
            distance = euclidean_distance(center, predicted)
            
            if distance < threshold and direction_ok(track, center):
                matches.append((distance, c_idx, t_idx))
    
    # 2. Sort by distance (best first)
    matches.sort(key=lambda x: x[0])
    
    # 3. Greedy assignment
    matched_centers = set()
    matched_tracks = set()
    assignments = []
    
    for dist, c_idx, t_idx in matches:
        if c_idx not in matched_centers and t_idx not in matched_tracks:
            assignments.append((c_idx, t_idx))
            matched_centers.add(c_idx)
            matched_tracks.add(t_idx)
    
    return assignments, matched_centers, matched_tracks
```

### 5.3 Velocity Prediction

```python
def predict_position(track):
    if len(track['path']) < 2:
        return track['path'][-1]
    
    # Linear velocity
    dx = track['path'][-1][0] - track['path'][-2][0]
    dy = track['path'][-1][1] - track['path'][-2][1]
    
    # Apply velocity decay for DTL view
    if is_dtl:
        dx *= 0.98
        dy *= 0.98
    
    # Add gravity (downward acceleration)
    predicted_x = track['path'][-1][0] + dx
    predicted_y = track['path'][-1][1] + dy + GRAVITY_ACCEL
    
    return (predicted_x, predicted_y)
```

### 5.4 Bounce Detection

```python
def has_bounced(path, consistent_frames=3, min_speed=2):
    """Detect surface contact via velocity reversal"""
    
    if len(path) < consistent_frames + 2:
        return False
    
    recent = path[-(consistent_frames + 2):]
    
    # Calculate velocity deltas
    dxs = [recent[i+1][0] - recent[i][0] for i in range(len(recent)-1)]
    dys = [recent[i+1][1] - recent[i][1] for i in range(len(recent)-1)]
    
    # Check for consistent direction then reversal
    def reversed_after_consistent(deltas, was_positive):
        significant = [d for d in deltas if abs(d) >= min_speed]
        if len(significant) < consistent_frames:
            return False
        
        prior, last = significant[:-1], significant[-1]
        
        if was_positive:
            # Was moving positive, now negative
            consistent = sum(d > 0 for d in prior) >= len(prior) - 1
            reversed = last < -min_speed
            return consistent and reversed
        else:
            # Was moving negative, now positive
            consistent = sum(d < 0 for d in prior) >= len(prior) - 1
            reversed = last > min_speed
            return consistent and reversed
    
    # Ground bounce: falling (dy>0) then rising (dy<0)
    ground_bounce = reversed_after_consistent(dys, was_positive=True)
    
    # Wall bounce: horizontal reversal
    wall_bounce = (reversed_after_consistent(dxs, was_positive=True) or
                   reversed_after_consistent(dxs, was_positive=False))
    
    return ground_bounce or wall_bounce
```

### 5.5 Direction Continuity Validation

```python
def direction_ok(path, new_pos, max_angle=120):
    if len(path) < 2:
        return True
    
    # For DTL, skip validation
    if is_dtl:
        return True
    
    # Previous velocity vector
    vx = path[-1][0] - path[-2][0]
    vy = path[-1][1] - path[-2][1]
    
    # New velocity vector
    dx = new_pos[0] - path[-1][0]
    dy = new_pos[1] - path[-1][1]
    
    # Check for zero vectors
    if (vx == 0 and vy == 0) or (dx == 0 and dy == 0):
        return True
    
    # Calculate denominator safely
    denom = sqrt(vx² + vy²) × sqrt(dx² + dy²)
    if denom < 1e-6:  # Numerical stability
        return True
    
    # Dot product and angle
    cos_angle = (vx×dx + vy×dy) / denom
    angle_deg = arccos(clip(cos_angle, -1, 1)) × 180/π
    
    return angle_deg < max_angle
```

---

## 6. User Interface

### 6.1 Sidebar Controls

**Phase 1: Detection** (Requires Rerun):
- Camera View: Radio ["Side View", "Down-the-Line (DTL)"]
- Shape Sensitivity: Slider [0.1-1.0, step 0.05, default 0.6]
- Yellow Threshold: Slider [50-255, default 120]

**Phase 2: Analysis** (Updates Instantly):
- Launch Zone Time Window: Slider [10-50%, step 5%, default 20%]
- Launch Angle Range: Range slider [-45° to 120°, default 20-80°]
- Max Height Range: Range slider [0-1200px, default 0-1200]
- Min Velocity: Slider [0-100 px/s, step 5, default 10]
- Target Height: Slider/Number [0-100%, default 40%]
- IQR Outlier Removal: Checkbox [default: False]
- IQR Threshold: Slider [1.5-4.0, step 0.5, default 2.5]

**Actions**:
- Reset All Data: Button (clears session state)
- Rerun Analysis: Button (restarts from video)

### 6.2 Main Display

**Top Row**: Video feed (left) + Trajectory chart (right)

**Phase Indicators**:
- 📤 Ready - Upload video to begin
- 🎬 Phase 1: Video Analysis Running
- 📊 Analysis Phase - Adjust sliders to refine

**Trajectory Chart**:
- Dark background (#1e1e1e)
- Rainbow-colored trajectories (golden ratio spacing)
- Launch zone overlay (cyan circle)
- Ball numbers at trajectory endpoints
- Updates every 10 frames during Phase 1
- Auto-updates on filter changes in Phase 2

### 6.3 Summary Section

**Statistics Table**:
- Metrics: Velocity, Launch Angle, Max Height, Target Distance
- Columns: Metric, Avg, Std Dev

**Ball Log Table**:
- Columns: Ball, Launch Time, Velocity, Angle, Height, Target X, Target Distance, Extrap
- Sorted by launch time
- Sequential ball numbers

**Distribution Plots** (if ≥3 balls):
- Launch Angle histogram with mean/median lines
- Max Height histogram with mean/median lines

**Trend Charts** (if ≥2 balls):
- Velocity over time with ball number labels
- Angle over time
- Height over time
- Mean lines overlaid

### 6.4 Target Accuracy Section

**Tabs**: "📊 Actual Trajectories" | "🔮 Projected (Physics Fit)"

**Top Chart** (Interactive Plotly):
- Autoscaled to trajectory bounds (10% padding)
- Actual trajectories (colored dots)
- Physics projection (dashed lines) - Projected tab only
- Fit region (green solid, width 4px) - Projected tab only
- Apex markers (X symbols)
- Target line (red dashed)
- Intercepts (yellow/green circles)
- Click to set target height (Actual tab only)

**Bottom Chart** (Matplotlib):
- X position vs Launch Time scatter
- Mean line (green)
- ±1σ bands (yellow dashed)
- Ball number labels

**Correlation Analysis** (if ≥3 balls):
- 4 metric cards with correlation coefficients
- 2×2 scatter plot grid with trend lines
- Interpretation guidance

### 6.5 Export Section

**Save Data Button**:
- CSV: Ball log with accuracy data
- JPG: Trajectory chart (150 DPI)
- Filename includes: video name, filter settings, timestamp

**Save PDF Button**:
- Page 1: Trajectories + 6 detection snapshots
- Page 2: Summary statistics + distributions
- Page 3: Trend charts over time
- Page 4: Target accuracy analysis
- Page 5: Correlation scatter plots (if available)
- Metadata: Title, Author, Subject, CreationDate

---

## 7. Configuration Parameters

### 7.1 Constants

```python
MIN_BALL_AREA = 100                    # Minimum contour area (px²)
MIN_TRAJECTORY_LENGTH = 5              # Minimum points for valid trajectory
MAX_FLIGHT_TIME_SEC = 3                # Maximum tracking duration (seconds)
GRAPH_UPDATE_INTERVAL = 10             # Frames between UI updates
MIN_VELOCITY = 10                      # Default minimum velocity (px/s)
SPATIAL_FILTER_MULTIPLIER = 3.0        # Launch zone radius multiplier
MIN_LAUNCH_ZONE_RADIUS = 50            # Minimum launch zone size (px)
MAX_LAUNCH_ZONE_RADIUS = 200           # Maximum launch zone size (px)
GRAVITY_ACCEL = 0.5                    # Default gravity (px/frame²) - overridden by fit
```

### 7.2 Fixed Parameters (Internal)

```python
match_threshold = 80                   # Pixels for track-center matching
memory_frames = 15                     # Frames to remember lost tracks
outlier_sensitivity = 3.0              # Launch zone filter sensitivity (unused)
speed = 1.0                           # Playback speed (unused)
auto_fps = True                       # Auto-detect FPS from video
manual_fps = 30                       # Fallback FPS if detection fails
```

### 7.3 DTL View Multipliers

```python
effective_match_threshold = match_threshold × 6.0     # 480px for DTL
effective_min_ball_area = MIN_BALL_AREA × 0.3        # 30 for DTL
effective_sensitivity = sensitivity × 0.4             # Reduced for DTL
effective_memory_frames = memory_frames × 3          # 45 frames for DTL
```

### 7.4 Color Detection Ranges

```python
lower_yellow = [20, sat_val, 100]      # HSV lower bound
upper_yellow = [35, 255, 255]          # HSV upper bound
```

---

## 8. Performance Requirements

### 8.1 Processing Speed

**Phase 1 (Video Analysis)**:
- Target: Process all frames (no frame skipping)
- Typical: 10-30ms per frame (1920×1080 @ 30fps)
- Breakdown:
  - CV operations: 5-10ms
  - Tracking logic: 2-5ms
  - UI updates (every 10 frames): 50-100ms

**Phase 2 (Filtering)**:
- Target: <100ms for filter updates
- Typical: 10-50ms (for 100 trajectories)
- Enables real-time slider interaction

### 8.2 Memory Usage

**Active Tracking**:
- ~10KB per active track (path data)
- Maximum ~50 concurrent tracks expected

**RAW Data Storage**:
- ~50KB per finalized trajectory (1000 points)
- ~5KB per ball log entry
- Typical video: 20-50 trajectories = 1-2.5MB

**Session State**:
- Total: 5-10MB for typical session
- No memory cleanup between videos (user must reset)

### 8.3 Scalability Limits

**Video Resolution**: Tested up to 1920×1080
**Frame Rate**: Tested up to 120 fps
**Video Duration**: Up to 60 seconds (1800 frames @ 30fps)
**Concurrent Balls**: Up to 50 simultaneous
**Total Trajectories**: Up to 500 per session

---

## 9. Export Formats

### 9.1 CSV Format

```csv
Ball,Launch Time (s),Initial Velocity (px/s),Launch Angle (°),Max Height (px from top),Target X (px),Target Distance (px),Extrapolated
1,0.53,245.3,45.2,123,856.2,12.3,No
2,1.21,238.7,43.8,118,870.5,1.8,Yes
...
```

**Columns**:
- Ball: Sequential number (chronological order)
- Launch Time (s): Timestamp
- Initial Velocity (px/s): Speed
- Launch Angle (°): Angle
- Max Height (px from top): Apex Y coordinate
- Target X (px): Landing position at target height
- Target Distance (px): Distance from mean landing position
- Extrapolated: Yes/No/N/A

### 9.2 JPG Format

**Content**: Trajectory chart with all filtered trajectories
**Resolution**: 150 DPI
**Background**: Dark (#1e1e1e)
**Dimensions**: Based on figure size (6×5 inches default)

### 9.3 PDF Format

**Page 1 - Detection Overview**:
- Top 45%: Trajectory chart with launch zone
- Bottom 45%: 6 detection snapshots in 2×3 grid

**Page 2 - Summary Statistics**:
- Text summary: Total balls, camera view, filter counts, statistics
- 3 histograms: Launch angle, max height, initial velocity

**Page 3 - Trend Charts**:
- 3 stacked plots: Velocity, Angle, Height vs Time
- Mean lines overlaid

**Page 4 - Target Accuracy**:
- Top: Trajectories with target line and intercepts
- Bottom: Accuracy distribution over time
- Autoscaled axes

**Page 5 - Correlation Analysis** (if ≥3 balls):
- 2×2 subplot grid: 4 correlation scatter plots
- Ball number labels
- Trend lines for significant correlations

**Metadata**:
- Title: "Ball Tracking Analysis Report"
- Author: "Ball Tracker Dashboard"
- Subject: "Trajectory Analysis and Accuracy Metrics"
- CreationDate: Timestamp

---

## 10. Future Enhancements

### 10.1 Planned Features

**High Priority**:
1. Multi-color ball detection (not just yellow)
2. Adaptive color calibration per video
3. Video codec compatibility improvements
4. Batch processing mode (multiple videos)

**Medium Priority**:
5. Real-time mode (webcam input)
6. Ball spin detection from trajectory curvature
7. Wind correction model
8. 3D trajectory reconstruction (stereo cameras)

**Low Priority**:
9. Machine learning ball detector (vs HSV threshold)
10. Automated launch zone detection (no manual params)
11. Export to Excel with charts
12. Cloud storage integration

### 10.2 Code Quality Improvements

**Refactoring**:
- Modularize into separate files (tracking.py, filtering.py, analysis.py, ui.py)
- Extract constants to config.py
- Add type hints throughout
- Create unit tests for algorithms
- Add integration tests for end-to-end flows

**Documentation**:
- Add docstrings to all functions
- Create architecture diagrams
- Document algorithm derivations
- Add code examples for key functions

**Performance**:
- Profile and optimize hot paths
- Consider numba JIT compilation for tracking loops
- Cache filter results to avoid recomputation
- Implement progressive rendering for large datasets

### 10.3 Known Issues

1. **Memory leak potential**: Matplotlib figure reused globally (intentional for performance)
2. **OpenCV version compatibility**: Handled but requires -2 index workaround
3. **Very small IQR**: Warning shown but filter still applied
4. **DTL accuracy**: Physics model may be less accurate for extreme DTL angles
5. **No undo**: User must rerun analysis to try different detection settings

---

## Appendix A: Coordinate Systems

### Image Coordinates
- Origin: Top-left corner
- X-axis: Left → Right (0 to width)
- Y-axis: Top → Down (0 to height)
- Apex: Minimum Y value (highest physical point)

### Physics Equations (Image Coordinates)
```
x(t) = x0 + vx × t
y(t) = y0 + vy0 × t + 0.5 × g_eff × t²
```
- g_eff > 0: Points downward (increases Y)
- vy0 < 0: Upward initial velocity (decreases Y)
- At apex: dy/dt = 0 → t_apex = -vy0 / g_eff

---

## Appendix B: Glossary

**Apex**: Highest point of trajectory (minimum Y coordinate)
**CEP**: Circular Error Probable - radius containing 50% of shots
**DTL**: Down-the-Line camera view (camera behind/in front of trajectory)
**HSV**: Hue, Saturation, Value color space
**IQR**: Interquartile Range (Q3 - Q1)
**Magnus Effect**: Lift force from ball spin (backspin creates upward force)
**R95**: Radius containing 95% of shots
**g_eff**: Effective gravity (includes true gravity + Magnus + drag)

---

## Appendix C: File Structure

```
ball-tracker/
├── app.py                    # Main application (1779 lines)
├── requirements.txt          # Python dependencies
├── .gitignore               # Excludes temp videos, exports
├── SPECIFICATION.md         # This document
└── .claude/
    └── settings.local.json  # Claude Code permissions
```

---

**Document Control**

| Version | Date       | Author        | Changes                           |
|---------|------------|---------------|-----------------------------------|
| 1.0     | 2026-04-05 | Claude + User | Initial specification document    |

---

*This specification is the source of truth for the Ball Tracker application. All code changes should align with this spec. When adding new features, update this document first.*
