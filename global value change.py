import multiprocessing
import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def update_global_value(shared_value, lock):
    with lock:
        shared_value.value += 1
        print(f"Global value updated to: {shared_value.value}")

def process_camera(camera_num, stop_event, shared_value, lock):
    print(f"Camera {camera_num} starting")
    upPointMark = mp_pose.PoseLandmark.LEFT_HIP
    middlePointMark = mp_pose.PoseLandmark.LEFT_KNEE
    lowPointMark = mp_pose.PoseLandmark.LEFT_ANKLE

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
                upPoint = [landmarks[upPointMark.value].x, landmarks[upPointMark.value].y]
                middlePoint = [landmarks[middlePointMark.value].x, landmarks[middlePointMark.value].y]
                lowPoint = [landmarks[lowPointMark.value].x, landmarks[lowPointMark.value].y]

                calculated_angle = calculate_angle(upPoint, middlePoint, lowPoint)

                # Display shared global value
                with lock:
                    current_global_value = shared_value.value

                cv2.putText(image, f"Global Value: {current_global_value}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

        with multiprocessing.Pool(processes=4) as pool:
            pool.starmap(
                process_camera,
                [(i, stop_event, shared_value, lock) for i in range(1, 5)]
            )

    print("All cameras completed.")
    cv2.destroyAllWindows()
