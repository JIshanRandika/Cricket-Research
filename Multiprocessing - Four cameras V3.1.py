import multiprocessing
import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define pairs of landmarks for angles
ANGLE_LANDMARKS = [
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('LEFT_HIP', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE'),
    ('LEFT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'LEFT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'LEFT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_SHOULDER'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_SHOULDER')
]

def update_global_value(shared_value, lock):
    with lock:
        shared_value.value += 1
        print(f"Global value updated to: {shared_value.value}")

def process_camera(camera_num, stop_event, shared_value, lock, dataframes):
    print(f"Camera {camera_num} starting")
    cap = cv2.VideoCapture(camera_num - 1)
    if not cap.isOpened():
        print(f"Camera {camera_num} not found.")
        return None

    fps = 30
    cap.set(cv2.CAP_PROP_FPS, fps)

    size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f'output_camera_{camera_num}.mp4'
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)
    angle_values = []
    start_time = time.time()
    frame_number = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angle_data = {'Time': time.time() - start_time}

                for angle_name, (up, mid, low) in enumerate(ANGLE_LANDMARKS):
                    upPoint = [landmarks[getattr(mp_pose.PoseLandmark, up).value].x, 
                               landmarks[getattr(mp_pose.PoseLandmark, up).value].y]
                    middlePoint = [landmarks[getattr(mp_pose.PoseLandmark, mid).value].x, 
                                   landmarks[getattr(mp_pose.PoseLandmark, mid).value].y]
                    lowPoint = [landmarks[getattr(mp_pose.PoseLandmark, low).value].x, 
                                landmarks[getattr(mp_pose.PoseLandmark, low).value].y]

                    calculated_angle = calculate_angle(upPoint, middlePoint, lowPoint)
                    angle_data[f'{angle_name}_Camera{camera_num}'] = calculated_angle

                    cv2.putText(image, f"{angle_name}:{calculated_angle:.2f}",
                                tuple(np.multiply(middlePoint, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                angle_values.append(angle_data)

                output_dir = f"frames_camera_{camera_num}"
                if not os.path.exists(output_dir):
                    save_data(angle_values, camera_num, shared_value.value, dataframes, lock)
                    os.makedirs(output_dir)
                    frame_number = 0
                    angle_values = []

                frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, image)
                frame_number += 1

            cv2.imshow(f'Camera {camera_num}', image)
            out.write(image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            elif key == ord(' '):
                update_global_value(shared_value, lock)

    cap.release()
    out.release()
    cv2.destroyWindow(f'Camera {camera_num}')
    print(f"Camera {camera_num} completed")

def save_data(angle_values, camera_num, current_global_value, dataframes, lock):
    df = pd.DataFrame(angle_values)
    excel_filename = f'Angles_Camera{camera_num}_V{current_global_value}.xlsx'
    try:
        df.to_excel(excel_filename, index=False)
        print(f"Excel file for Camera {camera_num} saved successfully.")
        with lock:
            dataframes.append(df)
    except Exception as e:
        print(f"Error saving Excel file for Camera {camera_num}:", e)

    if len(dataframes) == 3:
        save_combined_data(dataframes, current_global_value, lock)
        with lock:
            dataframes[:] = []

def save_combined_data(dataframes, prev_global_value, lock):
    with lock:
        combined_df = pd.concat(dataframes, axis=1)
        combined_filename = f'Combined_Angles_V{prev_global_value}.xlsx'
        combined_df.to_excel(combined_filename, index=False)
        print(f"Combined Excel file saved successfully: {combined_filename}")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        stop_event = manager.Event()
        shared_value = manager.Value('i', 0)  # Shared integer value
        lock = manager.Lock()  # Multiprocessing lock
        dataframes = manager.list()  # Shared list for dataframes

        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_camera,
                [(i, stop_event, shared_value, lock, dataframes) for i in range(1, 4)]
            )

    print("All cameras completed.")
    cv2.destroyAllWindows()
