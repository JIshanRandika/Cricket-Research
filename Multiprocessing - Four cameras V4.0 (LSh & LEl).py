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

def update_global_value(shared_value, lock):
    with lock:
        shared_value.value += 1
        print(f"Global value updated to: {shared_value.value}")

def process_camera(camera_num, stop_event, shared_value, lock, dataframes):
    print(f"Camera {camera_num} starting")
    leftHip = mp_pose.PoseLandmark.LEFT_HIP
    leftShoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
    leftElbow = mp_pose.PoseLandmark.LEFT_ELBOW
    leftWrist = mp_pose.PoseLandmark.LEFT_WRIST

    cap = cv2.VideoCapture(camera_num - 1)
    if not cap.isOpened():
        print(f"Camera {camera_num} not found.")
        return None

    fps = 30
    cap.set(cv2.CAP_PROP_FPS, fps)
    # with lock:
    #     current_global_value = shared_value.value
            
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
                leftHipPoint = [landmarks[leftHip.value].x, landmarks[leftHip.value].y]
                leftShoulderPoint = [landmarks[leftShoulder.value].x, landmarks[leftShoulder.value].y]
                leftElbowPoint = [landmarks[leftElbow.value].x, landmarks[leftElbow.value].y]
                leftWristPoint = [landmarks[leftWrist.value].x, landmarks[leftWrist.value].y]

                leftShoulder_angle = calculate_angle(leftHipPoint, leftShoulderPoint, leftElbowPoint)
                leftElbow_angle = calculate_angle(leftShoulderPoint, leftElbowPoint, leftWristPoint)

                cv2.putText(image, str(leftShoulder_angle),
                           tuple(np.multiply(leftShoulderPoint, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)
                cv2.putText(image, str(leftElbow_angle),
                           tuple(np.multiply(leftElbowPoint, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)
                
                # Display shared global value
                with lock:
                    current_global_value = shared_value.value

                cv2.putText(image, f"Global Value: {current_global_value}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                angle_values.append({
                    'Time': time.time() - start_time, 
                    f'LShoulder_Camera_{camera_num}': leftShoulder_angle,
                    f'LElbow_Camera_{camera_num}': leftElbow_angle
                    })

                output_dir = f"frames_camera_{camera_num}_{current_global_value}"
                if not os.path.exists(output_dir):
                    save_data(angle_values, camera_num, current_global_value, dataframes, lock)
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

# dataframes = []
def save_data(angle_values, camera_num, current_global_value, dataframes, lock):
    # global prev_value  # Use the global variable to keep track of the last angle value
    # global dataframes
    df = pd.DataFrame(angle_values)
    excel_filename = f'LSh_LEl_{camera_num}_0.75_Multiprocessing_{current_global_value}.xlsx'
    try:
        df.to_excel(os.path.join(os.path.dirname(__file__), excel_filename), index=False)
        print(f"Excel file for Camera {camera_num} saved successfully.")
        with lock:
            dataframes.append(df)
    except Exception as e:
        print(f"Error saving Excel file for Camera {camera_num}:", e)
    # print("new")
    # print(current_global_value)
    # print(len(dataframes))
    # if prev_value is not None:  # Check if there was a previous value
    if len(dataframes) == 3:
        save_combined_data(dataframes, current_global_value, lock)
        with lock:
            dataframes[:] = []


    print(f"Camera {camera_num}_{current_global_value} completed")

def save_combined_data(dataframes, prev_global_value, lock):
    print("saving_combined_data")
    with lock:
    # Combine results from all four cameras - Concat
        if len(dataframes) >= 3:  # Ensure at least 3 dataframes are present
                combined_df = pd.concat(dataframes, axis=1)
                try:
                    combined_df.to_excel(os.path.join(os.path.dirname(__file__), f'combined_Lsh_LEl_angles_Multiprocessing_concat_{prev_global_value}.xlsx'), index=False)
                    print("Combined Excel file saved successfully.")
                except Exception as e:
                    print("Error saving combined Excel file:", e)
                
        # Combine results from all four cameras - BasedonTime
        combined_dfT = dataframes[0]
        for df in dataframes[1:]:
            combined_dfT = pd.merge_asof(combined_dfT, df, on='Time', direction='nearest')
        
        # Add a new column for the average value
        try:
            # angle_columns = [col for col in combined_dfT.columns if col != 'Time']  # Exclude the 'Time' column
            # combined_dfT['Average'] = combined_dfT[angle_columns].mean(axis=1)
            # Extract columns related to LShoulder and LElbow
            lshoulder_columns = [col for col in combined_dfT.columns if 'LShoulder' in col]
            lelbow_columns = [col for col in combined_dfT.columns if 'LElbow' in col]

            # Calculate the average for LShoulder and LElbow separately
            combined_dfT['LShoulder_Avg'] = combined_dfT[lshoulder_columns].mean(axis=1)
            combined_dfT['LElbow_Avg'] = combined_dfT[lelbow_columns].mean(axis=1)
        except Exception as e:
            print("Error calculating the average column:", e)

        try:
            # Save to Excel
            combined_dfT.to_excel(os.path.join(os.path.dirname(__file__), f'combined_Lsh_LEl_angles_Multiprocessing_BasedonTime_{prev_global_value}.xlsx'), index=False)
            print("Combined Excel file saved successfully.")
        except Exception as e:
            print("Error saving combined Excel file:", e)

        try:
            # Create a new DataFrame with only 'Time' and 'Average'
            time_and_average_df = combined_dfT[['Time', 'LShoulder_Avg','LElbow_Avg']]
            
            # Save to Excel
            time_and_average_path = os.path.join(os.path.dirname(__file__), f'time_and_average_{prev_global_value}.xlsx')
            time_and_average_df.to_excel(time_and_average_path, index=False)
            print(f"'Time' and 'Average' Excel file saved successfully.")
        except Exception as e:
            print("Error saving 'Time' and 'Average' Excel file:", e)

        # Plotting the line chart
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot each camera's data and the Average column
            for column in combined_dfT.columns:
                if column != 'Time':  # Exclude the Time column
                    plt.plot(combined_dfT['Time'], combined_dfT[column], label=column)
            
            plt.xlabel('Time')
            plt.ylabel('Angle')
            plt.title('Knee Angles Over Time by Camera')
            plt.legend()
            plt.grid(True)
            
            # Save the chart as PNG
            chart_path = os.path.join(os.path.dirname(__file__), f'combined_Lsh_LEl_angles_Multiprocessing_BasedonTime_chart_{prev_global_value}.png')
            plt.savefig(chart_path)
            plt.show()
            print(f"Chart saved successfully at {chart_path}.")
        except Exception as e:
            print("Error generating or saving the chart:", e)

        try:
            # Plot the Time vs Average line chart
            plt.figure(figsize=(12, 6))
            plt.plot(time_and_average_df['Time'], time_and_average_df['LShoulder_Avg'], label='LShoulder_Avg', color='blue')
            plt.plot(time_and_average_df['Time'], time_and_average_df['LElbow_Avg'], label='LElbow_Avg', color='green')
            plt.xlabel('Time')
            plt.ylabel('Average Angle')
            plt.title('Average Knee Angle Over Time')
            plt.legend()
            plt.grid(True)
            
            # Save the chart as PNG
            time_and_average_chart_path = os.path.join(os.path.dirname(__file__), f'time_and_average_chart_{prev_global_value}.png')
            plt.savefig(time_and_average_chart_path)
            plt.show()
            print(f"'Time' and 'Average' chart saved successfully at.")
        except Exception as e:
            print("Error generating or saving the 'Time' and 'Average' chart:", e)


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
