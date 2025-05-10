import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import IsolationForest
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define actions for labeling
# Define actions for labeling
ACTIONS = [
    "standing", "sitting", "waving", "jumping", "squatting", 
    "running", "falling", "clapping", "pointing", "crouching",
    "fighting", "pushing", "loitering", "fainting", 
    "carrying_suspicious_object", "crowd_panic", "climbing", "lying_down"
]

# Define colors for visualization
ACTION_COLORS = {
    "standing": (0, 255, 0),              # Green
    "sitting": (255, 0, 0),               # Blue
    "waving": (0, 0, 255),                # Red
    "jumping": (255, 255, 0),             # Cyan
    "squatting": (255, 0, 255),           # Magenta
    "running": (0, 255, 255),             # Yellow
    "falling": (255, 165, 0),             # Orange
    "clapping": (128, 0, 128),            # Purple
    "pointing": (0, 128, 128),            # Teal
    "crouching": (128, 128, 0),           # Olive
    "fighting": (255, 0, 127),            # Pink
    "pushing": (127, 255, 0),             # Lime
    "loitering": (0, 127, 255),           # Sky Blue
    "fainting": (255, 99, 71),            # Tomato
    "carrying_suspicious_object": (75, 0, 130),  # Indigo
    "crowd_panic": (255, 20, 147),        # Deep Pink
    "climbing": (0, 255, 127),            # Spring Green
    "lying_down": (139, 69, 19),          # Saddle Brown
    "anomaly": (0, 0, 0)                  # Black
}

# Function to extract keypoints
def extract_keypoints(landmarks):
    keypoints = []
    if landmarks:
        for lm in landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.visibility])
    return keypoints

# Function to extract features for action recognition
def extract_action_features(keypoints, prev_keypoints=None, energy_window=None):
    if not keypoints or len(keypoints) < 99:
        return None
        
    features = []
    
    # Existing features (head, shoulders, hips, knees, ankles, wrists, elbows)
    head_x, head_y = keypoints[0], keypoints[1]
    left_shoulder_x, left_shoulder_y = keypoints[33], keypoints[34]
    right_shoulder_x, right_shoulder_y = keypoints[21], keypoints[22]
    left_hip_x, left_hip_y = keypoints[69], keypoints[70]
    right_hip_x, right_hip_y = keypoints[57], keypoints[58]
    left_knee_x, left_knee_y = keypoints[81], keypoints[82]
    right_knee_x, right_knee_y = keypoints[75], keypoints[76]
    left_ankle_x, left_ankle_y = keypoints[87], keypoints[88]
    right_ankle_x, right_ankle_y = keypoints[93], keypoints[94]
    left_wrist_x, left_wrist_y = keypoints[45], keypoints[46]
    right_wrist_x, right_wrist_y = keypoints[51], keypoints[52]
    left_elbow_x, left_elbow_y = keypoints[39], keypoints[40]
    right_elbow_x, right_elbow_y = keypoints[27], keypoints[28]
    
    # Calculate shoulder width for normalization
    shoulder_width = np.sqrt((right_shoulder_x - left_shoulder_x)**2 + 
                           (right_shoulder_y - left_shoulder_y)**2)
    if shoulder_width == 0:
        return None
    
    # New features for anomalous actions
    # 1. Fighting: Rapid arm movements
    arm_movement = np.sqrt((left_wrist_x - left_elbow_x)**2 + (left_wrist_y - left_elbow_y)**2) + \
                   np.sqrt((right_wrist_x - right_elbow_x)**2 + (right_wrist_y - right_elbow_y)**2)
    
    # 2. Pushing: Hand near another person's torso
    hand_torso_distance = np.sqrt((left_wrist_x - (left_hip_x + right_hip_x)/2)**2) + \
                          np.sqrt((left_wrist_y - (left_hip_y + right_hip_y)/2)**2)
    
    # 3. Loitering: Low movement energy over time
    movement_energy = calculate_movement_energy(keypoints, prev_keypoints) if prev_keypoints else 0
    
    # 4. Fainting: Sudden downward movement
    hip_pos = (left_hip_y + right_hip_y) / 2
    
    # 5. Carrying Suspicious Objects: Hand positions near a large object
    hand_object_distance = np.sqrt((left_wrist_x - right_wrist_x)**2 + (left_wrist_y - right_wrist_y)**2)
    
    # 6. Crowd Panic: High movement energy in multiple people
    crowd_energy = np.mean(energy_window) if energy_window and len(energy_window) > 0 else 0
    
    # 7. Climbing: High vertical movement
    vertical_movement = np.abs(head_y - left_ankle_y)
    
    # 8. Lying Down: Low vertical position for an extended period
    lying_down_ratio = (left_hip_y + right_hip_y) / 2
    
    # Append new features
    features.extend([
        arm_movement,
        hand_torso_distance,
        movement_energy,
        hip_pos,
        hand_object_distance,
        crowd_energy,
        vertical_movement,
        lying_down_ratio
    ])
    
    return features
# Function to classify action based on pose keypoints
def classify_action(keypoints, prev_keypoints, energy_window=None, action_classifier=None):
    if not keypoints or len(keypoints) < 99:
        return "unknown"
        
    features = extract_action_features(keypoints, prev_keypoints, energy_window)
    if not features:
        return "unknown"
        
    # Use classifier if available
    if action_classifier is not None:
        try:
            prediction = action_classifier.predict([features])[0]
            return prediction
        except:
            pass
    
    # Rule-based classification for anomalous actions
    arm_movement, hand_torso_distance, movement_energy, hip_pos, hand_object_distance, crowd_energy, vertical_movement, lying_down_ratio = features
    
    # Fighting: Rapid arm movements
    if arm_movement > 0.5:
        return "fighting"
    
    # Pushing: Hand near another person's torso
    if hand_torso_distance < 0.2:
        return "pushing"
    
    # Loitering: Low movement energy over time
    if movement_energy < 0.1:
        return "loitering"
    
    # Fainting: Sudden downward movement
    if prev_keypoints:
        prev_hip_pos = extract_action_features(prev_keypoints, None, energy_window)[3]
        if hip_pos - prev_hip_pos > 0.3:
            return "fainting"
    
    # Carrying Suspicious Objects: Hands close together
    if hand_object_distance < 0.1:
        return "carrying_suspicious_object"
    
    # Crowd Panic: High movement energy in multiple people
    if crowd_energy > 1.0:
        return "crowd_panic"
    
    # Climbing: High vertical movement
    if vertical_movement > 0.8:
        return "climbing"
    
    # Lying Down: Low vertical position for an extended period
    if lying_down_ratio > 0.9:
        return "lying_down"
    
    # Default to standing
    return "standing"

# Function to calculate movement energy
def calculate_movement_energy(current_frame, prev_frame):
    if prev_frame is None or len(current_frame) != len(prev_frame):
        return 0
    
    # Calculate Euclidean distance between consecutive frames
    # Skip visibility values (every 3rd value)
    positions_only = []
    for i in range(0, len(current_frame), 3):
        if i+1 < len(current_frame):
            positions_only.extend([current_frame[i], current_frame[i+1]])
    
    prev_positions_only = []
    for i in range(0, len(prev_frame), 3):
        if i+1 < len(prev_frame):
            prev_positions_only.extend([prev_frame[i], prev_frame[i+1]])
    
    if len(positions_only) != len(prev_positions_only):
        return 0
        
    return np.sqrt(np.sum(np.square(np.array(positions_only) - np.array(prev_positions_only))))

# Main capture function
def capture_pose_data(duration=60, detect_anomalies=True):
    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check your camera connection.")
        return None, None
    
    # Define Column Names (Keypoints)
    columns = [f"{pt}_{axis}" for pt in range(33) for axis in ['X', 'Y', 'Visibility']]
    
    # Data storage
    pose_data = []
    action_labels = []
    timestamps = []
    frame_features = []
    
    # For manual and automatic action labeling
    user_selected_action = "standing"  # Default action from user input
    detected_action = "standing"       # Action detected automatically
    
    print(f"Capturing video for {duration} seconds... Press 'q' to stop.")
    print("Available actions: " + ", ".join(ACTIONS))
    print("Press keys 1-5 to change the current action for training data")
    print("The system will also automatically detect your actions")
    
    start_time = time.time()
    prev_keypoints = None
    frame_count = 0
    
    # For measuring movement energy
    energy_window = []
    action_timestamps = []
    
    # For anomaly detection during capture
    energy_history = []
    is_anomaly = False
    anomaly_count = 0
    
    # Action classifier
    action_classifier = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break
        
        frame_time = time.time() - start_time
            
        # Convert frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose detection
        result = pose.process(rgb_frame)
        
        # Extract keypoints
        keypoints = extract_keypoints(result.pose_landmarks)
        
        # Auto-detect action
        if keypoints:
            detected_action = classify_action(keypoints, prev_keypoints, energy_window, action_classifier)
        
        # Calculate movement energy
        energy = calculate_movement_energy(keypoints, prev_keypoints)
        energy_history.append(energy)
        
        # Real-time anomaly detection
        if len(energy_history) > 30:
            energy_history.pop(0)
            if len(energy_history) > 10:
                avg_energy = np.mean(energy_history[:-5])
                current_energy = np.mean(energy_history[-5:])
                energy_ratio = current_energy / (avg_energy + 0.01)  # Avoid division by zero
                
                # Detect sudden changes in movement energy as potential anomalies
                is_anomaly = energy_ratio > 3.0 or energy_ratio < 0.2
                if is_anomaly:
                    anomaly_count += 1
        
        # Display actions on frame
        cv2.putText(frame, f"User Action: {user_selected_action}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACTION_COLORS.get(user_selected_action, (0, 0, 255)), 2)
        cv2.putText(frame, f"Detected: {detected_action}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACTION_COLORS.get(detected_action, (0, 0, 255)), 2)
        
        # Show anomaly indicator
        if is_anomaly:
            cv2.putText(frame, "ANOMALY DETECTED!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save keypoints with timestamp and both action labels
        if keypoints:
            pose_data.append(keypoints)
            # Save both the user-selected and detected actions
            action_info = f"{user_selected_action}|{detected_action}|{is_anomaly}"
            action_labels.append(action_info)
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(current_timestamp)
            
            # Extract features for potential classifier training
            features = extract_action_features(keypoints, prev_keypoints, energy_window)
            if features:
                frame_features.append((features, user_selected_action))
            
            # Track movement energy for anomaly detection
            if len(keypoints) == 99:  # Ensure we have complete pose data
                energy_window.append(energy)
                if len(energy_window) > 30:  # Keep a rolling window
                    energy_window.pop(0)
        
        # Draw Pose Landmarks
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                result.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
        
        # Display time remaining
        time_left = max(0, duration - (time.time() - start_time))
        cv2.putText(frame, f"Time left: {time_left:.1f}s", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display video
        cv2.imshow("Real-Time Pose Capture", frame)
        
        # Check for key presses to change action
        key = cv2.waitKey(1) & 0xFF
        
        # Number keys 1-5 to change action
        if 49 <= key <= 53:  # ASCII codes for 1-5
            action_idx = key - 49  # Convert to 0-4 index
            if action_idx < len(ACTIONS):
                prev_action = user_selected_action
                user_selected_action = ACTIONS[action_idx]
                print(f"Changed user action from {prev_action} to {user_selected_action}")
                action_timestamps.append((time.time(), user_selected_action))
                
                # Try to train a classifier with collected data
                if len(frame_features) > 20:
                    try:
                        X = [f[0] for f in frame_features]
                        y = [f[1] for f in frame_features]
                        action_classifier = KNeighborsClassifier(n_neighbors=3)
                        action_classifier.fit(X, y)
                        print("Updated action classifier with new data")
                    except Exception as e:
                        print(f"Could not train classifier: {e}")
        
        # 'q' to quit
        if key == ord('q') or (time.time() - start_time) > duration:
            break
            
        prev_keypoints = keypoints
        frame_count += 1
        
    # Release camera
    cap.release()
    cv2.destroyAllWindows()
    
    # Create DataFrame
    if not pose_data:
        print("No pose data was captured!")
        return None, None
        
    df = pd.DataFrame(pose_data, columns=columns)
    df["Subject"] = 1
    
    # Split the combined action info into separate columns
    action_split = [a.split("|") for a in action_labels]
    df["UserAction"] = [a[0] for a in action_split]
    df["DetectedAction"] = [a[1] for a in action_split]
    df["RealTimeAnomaly"] = [a[2] == "True" for a in action_split]
    df["Timestamp"] = timestamps
    
    # Save as UCI HAR Format (adapting to include both action labels)
    filename = f"pose_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Pose data saved to {filename}")
    
    print(f"Detected {anomaly_count} potential anomalies during capture")
    
    return df, energy_window, action_timestamps
# Advanced anomaly detection function
def detect_anomalies(df, energy_window, action_timestamps):
    if df is None or df.empty:
        print("No data available for anomaly detection")
        return
        
    print("\n===== ADVANCED ANOMALY DETECTION =====")
    
    # Check if we have enough data
    if len(df) < 20:
        print("Not enough data for reliable anomaly detection")
        return
    
    # Extract features for anomaly detection
    pose_features = []
    for i in range(len(df)):
        row = df.iloc[i]
        pose_data = []
        for j in range(0, 99, 3):  # Get only X,Y coordinates, skip visibility
            if j+1 < 99:
                pose_data.extend([row[j], row[j+1]])
        pose_features.append(pose_data)
    
    # Normalize features for better detection
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pose_features)
    
    # Use Isolation Forest for pose anomaly detection
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(scaled_features)
    
    # Add anomaly predictions to dataframe
    df["IsAnomalousPose"] = [pred == -1 for pred in preds]
    
    # Detect action-specific anomalies
    for action in df["DetectedAction"].unique():
        action_df = df[df["DetectedAction"] == action]
        if len(action_df) >= 10:
            action_features = []
            for i in action_df.index:
                row = df.loc[i]
                pose_data = []
                for j in range(0, 99, 3):
                    if j+1 < 99:
                        pose_data.extend([row[j], row[j+1]])
                action_features.append(pose_data)
            
            # Detect anomalies within this specific action
            if len(action_features) >= 10:
                action_scaler = StandardScaler()
                scaled_action_features = action_scaler.fit_transform(action_features)
                
                action_clf = IsolationForest(contamination=0.1, random_state=42)
                action_preds = action_clf.fit_predict(scaled_action_features)
                
                # Map predictions back to original indices
                for i, (idx, pred) in enumerate(zip(action_df.index, action_preds)):
                    if pred == -1:
                        df.at[idx, "IsActionSpecificAnomaly"] = True
    
    # Ensure the column exists even if no action-specific anomalies were found
    if "IsActionSpecificAnomaly" not in df.columns:
        df["IsActionSpecificAnomaly"] = False
    
    # Flag anomalous actions as anomalies
    anomalous_actions = [
        "fighting", "pushing", "loitering", "fainting", 
        "carrying_suspicious_object", "crowd_panic", 
        "climbing", "lying_down"
    ]
    df["IsAnomalousAction"] = df["DetectedAction"].isin(anomalous_actions)
    
    # Combine all anomaly types
    df["IsAnomaly"] = (
        df["IsAnomalousPose"] | 
        df["IsActionSpecificAnomaly"] | 
        df["RealTimeAnomaly"] | 
        df["IsAnomalousAction"]
    )
    
    # Save results with anomaly information
    anomaly_file = f"pose_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(anomaly_file, index=False)
    
    # Count and report anomalies
    total_anomalies = df["IsAnomaly"].sum()
    pose_anomalies = df["IsAnomalousPose"].sum()
    action_anomalies = df["IsActionSpecificAnomaly"].sum()
    realtime_anomalies = df["RealTimeAnomaly"].sum()
    anomalous_action_count = df["IsAnomalousAction"].sum()
    
    print(f"\nANOMALY DETECTION RESULTS:")
    print(f"Total frames analyzed: {len(df)}")
    print(f"Total anomalies detected: {total_anomalies} ({total_anomalies/len(df)*100:.1f}%)")
    print(f"  - General pose anomalies: {pose_anomalies}")
    print(f"  - Action-specific anomalies: {action_anomalies}")
    print(f"  - Real-time detected anomalies: {realtime_anomalies}")
    print(f"  - Anomalous actions detected: {anomalous_action_count}")
    
    # Report anomalies by action
    print("\nAnomalies by Action Type:")
    for action in df["DetectedAction"].unique():
        action_df = df[df["DetectedAction"] == action]
        action_anomaly_count = action_df["IsAnomaly"].sum()
        print(f"  - {action}: {action_anomaly_count} anomalies out of {len(action_df)} frames ({action_anomaly_count/len(action_df)*100:.1f}%)")
    
    # Create a timeline of anomalies
    print("\nAnomaly Timeline (showing first 10):")
    anomaly_df = df[df["IsAnomaly"]]
    for i, (idx, row) in enumerate(anomaly_df.iterrows()):
        if i >= 10:
            print(f"... and {len(anomaly_df) - 10} more anomalies")
            break
        anomaly_time = row["Timestamp"]
        action = row["DetectedAction"]
        print(f"  - Frame {idx}: Anomalous {action} at {anomaly_time}")
    
    return df
def print_system_info():
    print("\n" + "="*50)
    print("REAL-TIME POSE DETECTION AND ANOMALY DETECTION SYSTEM")
    print("="*50)
    print("\nHow this system works:")
    print("1. POSE DETECTION:")
    print("   - Uses MediaPipe's pose detection model to track 33 body keypoints")
    print("   - Each keypoint has an X, Y position and visibility score (0-1)")
    print("   - Keypoints include face, shoulders, elbows, wrists, hips, knees, ankles")
    
    print("\n2. ACTION CLASSIFICATION:")
    print("   - Automatically detects actions based on body posture and movement")
    print("   - Uses both rule-based detection and machine learning (KNN)")
    print("   - Recognizes actions like standing, sitting, jumping, squatting, waving, running, falling, clapping, pointing, crouching")
    print("   - You can also manually label actions with keys 1-5 for training")
    
    print("\n3. ANOMALY DETECTION:")
    print("   - Real-time: Detects sudden changes in movement energy")
    print("   - Post-processing: Uses Isolation Forest algorithm")
    print("   - Identifies general pose anomalies across all actions")
    print("   - Finds action-specific anomalies within each action category")
    
    print("\n4. DATA FORMAT:")
    print("   - Saves data in UCI HAR (Human Activity Recognition) format")
    print("   - Includes raw pose keypoints, action labels, and anomaly flags")
    print("   - Can be used for further analysis or machine learning")
    
    print("\nREADY TO START CAPTURE")
    print("="*50 + "\n")

def main():
    print_system_info()
    
    # Ask user how long to capture
    duration = 60
    try:
        user_duration = input("Enter capture duration in seconds (default 60): ")
        if user_duration.strip():
            duration = int(user_duration)
    except:
        print("Invalid input. Using default duration of 60 seconds.")
    
    # Capture data
    df, energy_window, action_timestamps = capture_pose_data(duration=duration)
    
    # Detect anomalies
    if df is not None:
        anomaly_df = detect_anomalies(df, energy_window, action_timestamps)
        
        # Display summary
        print("\n===== CAPTURE SUMMARY =====")
        print(f"Total frames captured: {len(df)}")
        
        # Activity distribution
        print("\nActivity Distribution:")
        for action in ACTIONS:
            count = len(df[df["DetectedAction"] == action])
            percentage = count / len(df) * 100 if len(df) > 0 else 0
            print(f"  - {action}: {count} frames ({percentage:.1f}%)")
        
        print("\nAnomaly Summary:")
        anomaly_count = df["IsAnomaly"].sum()
        print(f"  - Total anomalies: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
        
        print("\nData saved for further analysis.")

if __name__ == "__main__":
    main()