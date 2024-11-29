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
                upPoint = [landmarks[upPointMark.value].x, landmarks[upPointMark.value].y]
                middlePoint = [landmarks[middlePointMark.value].x, landmarks[middlePointMark.value].y]
                lowPoint = [landmarks[lowPointMark.value].x, landmarks[lowPointMark.value].y]

                calculated_angle = calculate_angle(upPoint, middlePoint, lowPoint)

                cv2.putText(image, str(calculated_angle),
                           tuple(np.multiply(middlePoint, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)
                
                # Display shared global value
                with lock:
                    current_global_value = shared_value.value

                cv2.putText(image, f"Global Value: {current_global_value}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                angle_values.append({'Time': time.time() - start_time, f'LKnee_Camera{camera_num}': calculated_angle})

                output_dir = f"frames_camera_{camera_num}_{current_global_value}"
                if not os.path.exists(output_dir):
                    save_data(angle_values, camera_num, current_global_value)
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

# temp_data = []
# prev_value = None
dataframes = []
def save_data(angle_values, camera_num, current_global_value):
    # global prev_value  # Use the global variable to keep track of the last angle value
    global dataframes
    df = pd.DataFrame(angle_values)
    excel_filename = f'LeftKnee_V{camera_num}_0.75_Multiprocessing_{current_global_value}.xlsx'
    try:
        df.to_excel(os.path.join(os.path.dirname(__file__), excel_filename), index=False)
        print(f"Excel file for Camera {camera_num} saved successfully.")
        dataframes.append(df)
    except Exception as e:
        print(f"Error saving Excel file for Camera {camera_num}:", e)
    print("new")
    print(current_global_value)
    # print("Prev")
    # print(prev_value)
    # dataframes.append(df)
    # if prev_value is not None:  # Check if there was a previous value
    if len(dataframes) == 3:
    # if current_global_value > prev_value:
        save_combined_data(dataframes, current_global_value)
        dataframes = []
    # prev_value = current_global_value
    # prev_value = current_global_value

    # dataframes.append(df)

    # global temp_data
    
    # # Initialize the data list for the current global value if not present
    # if current_global_value not in temp_data:
    #     temp_data[current_global_value] = []
    
    # # Append the current data to the list
    # temp_data[current_global_value].append(pd.DataFrame(angle_values))
    # # Check if the current_global_value has changed
    # keys = sorted(temp_data.keys())  # Ensure the keys are in order

    # if len(keys) > 1:
    #     # The previous global value (last in sorted order before current)
    #     prev_global_value = keys[-2]
        
    #     # Save the data for the previous global value and clear it
    #     save_combined_data(temp_data[prev_global_value], prev_global_value)
    #     del temp_data[prev_global_value]  # Clear the stored data


    print(f"Camera {camera_num}_{current_global_value} completed")

def save_combined_data(dataframes, prev_global_value):
    print("save_combined_data")
    # Combine results from all four cameras - Concat
    if dataframes:
            combined_df = pd.concat(dataframes, axis=1)
            try:
                combined_df.to_excel(os.path.join(os.path.dirname(__file__), f'combined_knee_angles_Multiprocessing_concat_{prev_global_value}.xlsx'), index=False)
                print("Combined Excel file saved successfully.")
            except Exception as e:
                print("Error saving combined Excel file:", e)

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

        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_camera,
                [(i, stop_event, shared_value, lock) for i in range(1, 4)]
            )

    print("All cameras completed.")
    cv2.destroyAllWindows()
