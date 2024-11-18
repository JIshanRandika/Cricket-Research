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

def getUpPoint():
    return mp_pose.PoseLandmark.LEFT_HIP

def getMiddlePoint():
    return mp_pose.PoseLandmark.LEFT_KNEE

def getLowPoint():
    return mp_pose.PoseLandmark.LEFT_ANKLE

def calculate_angle1(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    return angle

def rescale_frame1(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def find_available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def process_camera(camera_num):
    print(f"Camera {camera_num} started")
    upPointMark = getUpPoint()
    middlePointMark = getMiddlePoint()
    lowPointMark = getLowPoint()
    
    available_cameras = find_available_cameras()
    if len(available_cameras) >= camera_num:
        if camera_num == 1:
            cap = cv2.VideoCapture(0)  # First video file
        elif camera_num == 2:
            cap = cv2.VideoCapture(1)  # Second video file
        else:
            cap = cv2.VideoCapture(2)  # Third video file
        print(f"Capturing Camera {camera_num}")
    else:
        print(f"Camera {camera_num} not found.")
        return None

    counter = 0
    stage = None
    angle_values = []
    size = (640, 480)
    fps = 30
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f'output_camera_{camera_num}.mp4'
    out = cv2.VideoWriter(os.path.join(os.path.dirname(__file__), output_filename), fourcc, fps, size)

    start_time = time.time()
    output_dir = f"frames_camera_{camera_num}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_number = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                upPoint = [landmarks[upPointMark.value].x, landmarks[upPointMark.value].y]
                middlePoint = [landmarks[middlePointMark.value].x, landmarks[middlePointMark.value].y]
                lowPoint = [landmarks[lowPointMark.value].x, landmarks[lowPointMark.value].y]

                calculated_angle = calculate_angle1(upPoint, middlePoint, lowPoint)

                cv2.putText(image, str(calculated_angle),
                           tuple(np.multiply(middlePoint, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)

                if calculated_angle > 169:
                    stage = "UP"
                if calculated_angle <= 90 and stage == 'UP':
                    stage = "DOWN"
                    counter += 1
                    print(f"Camera {camera_num} Counter: {counter}")

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                out.write(image)
                angle_values.append({'Time': time.time() - start_time, f'LKnee_Camera{camera_num}': calculated_angle})

                frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, image)

            except Exception as e:
                print(f"Error in Camera {camera_num}:", e)

            cv2.imshow(f'Camera {camera_num}', image)
            frame_number += 1
            time.sleep(0.13)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyWindow(f'Camera {camera_num}')
    
    df = pd.DataFrame(angle_values)
    excel_filename = f'LeftKnee_V{camera_num}_0.75.xlsx'
    try:
        df.to_excel(os.path.join(os.path.dirname(__file__), excel_filename), index=False)
        print(f"Excel file for Camera {camera_num} saved successfully.")
    except Exception as e:
        print(f"Error saving Excel file for Camera {camera_num}:", e)

    print(f"Camera {camera_num} completed")
    return df

def run_function(camera_num):
    return process_camera(camera_num)

if __name__ == '__main__':
    start_time = time.time()

    # Create a pool of worker processes for 3 cameras
    with multiprocessing.Pool(processes=3) as pool:
        # Start all three camera functions in parallel
        results = pool.map(run_function, [1, 2, 3])

    end_time = time.time()

    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Process results
    for i, result in enumerate(results, 1):
        if isinstance(result, pd.DataFrame):
            dataframes.append(result)
            print(f"DataFrame from Camera {i} added to the array.")
        else:
            print(f"Result from Camera {i} is not a DataFrame, skipped.")

    # Combine results from all three cameras
    if dataframes:
        combined_df = pd.concat(dataframes, axis=1)
        try:
            combined_df.to_excel(os.path.join(os.path.dirname(__file__), 'combined_knee_angles.xlsx'), index=False)
            print("Combined Excel file saved successfully.")
        except Exception as e:
            print("Error saving combined Excel file:", e)

    print(f"All cameras completed in {end_time - start_time:.2f} seconds")