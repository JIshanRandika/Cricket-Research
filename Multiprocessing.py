import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
import pandas as pd
import multiprocessing
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    vec1 = np.array(a) - np.array(b)
    vec2 = np.array(c) - np.array(b)
    radians = np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2))
    angle = np.degrees(radians)
    return angle

# Function to find cameras
def find_available_cameras():
    available_cameras = []
    for i in range(10):  # Try indexing from 0 to 9 (adjust as needed)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def function1():
    print("Function 1 started")
    time.sleep(2)  # Simulating some work
    available_cameras = find_available_cameras()
    if len(available_cameras) >= 2:
        cap1 = cv2.VideoCapture(available_cameras[0])
        print(f"Capturing from cameras {available_cameras[0]} and {available_cameras[1]}")
    else:
        print("Not enough cameras found.")

    counter1 = 0
    stage1 = None
    angle_values1 = []
    size1 = (640, 480)
    frame_rate1 = cap1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(os.path.join(os.path.dirname(__file__),'camera1_output.mp4'), fourcc, 5, size1)

    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap1.isOpened():
            elapsed_time = time.time() - start_time
            current_time = time.time() - start_time
            ret1, frame1 = cap1.read()
            if not ret1:
                break
            image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image1.flags.writeable = False
            results1 = pose.process(image1)
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            try:
                landmarks1 = results1.pose_landmarks.landmark
                upPoint = [landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                middlePoint = [landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lowPoint = [landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                calculated_angle = calculate_angle(upPoint, middlePoint, lowPoint)

                # Visualization and counter logic
                cv2.putText(image1, str(calculated_angle),
                            tuple(np.multiply(middlePoint, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)

                if calculated_angle > 169:
                    stage1 = "UP"
                if calculated_angle <= 90 and stage1 == 'UP':
                    stage1 = "DOWN"
                    counter1 += 1
                    print(counter1)

                # Drawing landmarks and connections
                mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                out1.write(image1)
                angle_values1.append({'Time': current_time, 'Camera 1 Angle': calculated_angle})
            except Exception as e:
                print("Error:", e)
            cv2.imshow('Camera 1', image1)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap1.release()
    cv2.destroyAllWindows()
    df1 = pd.DataFrame(angle_values1)
    try:
        df1.to_excel(os.path.join(os.path.dirname(__file__),'camera1_angles.xlsx'), index=False)
        print("Excel files saved successfully.")
    except Exception as e:
        print("Error saving Excel files:", e)

    try:
        out1.release()
        print("Video files saved successfully.")
    except Exception as e:
        print("Error saving Video files:", e)

    print("Function 1 completed")
    return "Result from Function 1"

def function2():
    print("Function 2 started")
    time.sleep(3)  # Simulating some work
    print("Function 2 completed")
    return "Result from Function 2"

def function3():
    print("Function 3 started")
    time.sleep(1)  # Simulating some work
    print("Function 3 completed")
    return "Result from Function 3"

def run_function(func):
    return func()

if __name__ == '__main__':
    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=3) as pool:
        # Start the functions in parallel
        results = pool.map(run_function, [function1, function2, function3])

    end_time = time.time()

    # Print results
    for i, result in enumerate(results, 1):
        print(f"Result from Function {i}: {result}")

    print(f"All functions completed in {end_time - start_time:.2f} seconds")