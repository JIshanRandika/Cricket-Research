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
    # return mp_pose.PoseLandmark.RIGHT_SHOULDER
    return mp_pose.PoseLandmark.LEFT_HIP

def getMiddlePoint():
    # return mp_pose.PoseLandmark.RIGHT_ELBOW
    return mp_pose.PoseLandmark.LEFT_KNEE

def getLowPoint():
    # return mp_pose.PoseLandmark.RIGHT_WRIST
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

# Function to calculate angle between three points
# def calculate_angle(a, b, c):
#     vec1 = np.array(a) - np.array(b)
#     vec2 = np.array(c) - np.array(b)
#     radians = np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2))
#     angle = np.degrees(radians)
#     return angle

# Function to find cameras
def find_available_cameras():
    available_cameras = []
    for i in range(10):  # Try indexing from 0 to 9 (adjust as needed)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def cameraOne():
    print("Camera One started")
    # time.sleep(2)  # Simulating some work
    upPointMark = getUpPoint()
    middlePointMark = getMiddlePoint()
    lowPointMark = getLowPoint()
    available_cameras = find_available_cameras()
    if len(available_cameras) >= 1:
        cap1 = cv2.VideoCapture("1.mp4")
        # desired_fps = 34  # For example, set to 10 FPS
        # cap1.set(cv2.CAP_PROP_FPS, desired_fps)

        # # Check if the frame rate was set successfully
        # fps = cap1.get(cv2.CAP_PROP_FPS)
        # print(f"Frame rate set to: {fps} FPS")

        # desired_fps = 50
        # frame_interval = 10

        print(f"Capturing Cameras One from {available_cameras[0]}")
    else:
        print("Not enough cameras found.")

    counter1 = 0
    stage1 = None
    angle_values1 = []
    size1 = (640, 480)
    fps = 30  # Set your desired frame rate 
    frame_rate1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(os.path.join(os.path.dirname(__file__),'output.mp4'), fourcc, fps, size1)

    start_time = time.time()
    output_dir = "frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_number = 0  # Initialize the frame counter

    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
        while cap1.isOpened():
            elapsed_time = time.time() - start_time
            current_time = time.time() - start_time
            ret1, frame1 = cap1.read()
            # if frame1 is not None:
            #     frame1_ = rescale_frame1(frame1, percent=75)
            if not ret1:
                break
            image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image1.flags.writeable = False
            results1 = pose.process(image1)
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            try:
                landmarks1 = results1.pose_landmarks.landmark
                upPoint = [landmarks1[upPointMark.value].x, landmarks1[upPointMark.value].y]
                middlePoint = [landmarks1[middlePointMark.value].x, landmarks1[middlePointMark.value].y]
                lowPoint = [landmarks1[lowPointMark.value].x, landmarks1[lowPointMark.value].y]

                calculated_angle = calculate_angle1(upPoint, middlePoint, lowPoint)

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
                angle_values1.append({'Time': time.time() - start_time, 'LElbow_MediaPipe': calculated_angle})

                # Save the image with frame number
                frame_filename = os.path.join(output_dir, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, image1)

            except Exception as e:
                print("Error:", e)
            cv2.imshow('Camera 1', image1)
            
            frame_number += 1  # Increment the frame counter

            # frame_interval = 10
            # elapsed_time = time.time() - start_time
            # time_to_wait = max(0, frame_interval - elapsed_time)
            time.sleep(0.13)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap1.release()
    cv2.destroyAllWindows()
    df1 = pd.DataFrame(angle_values1)
    try:
        df1.to_excel(os.path.join(os.path.dirname(__file__),'LeftKnee_V1_0.75.xlsx'), index=False)
        print("Excel files saved successfully.")
    except Exception as e:
        print("Error saving Excel files:", e)

    try:
        out1.release()
        print("Video files saved successfully.")
    except Exception as e:
        print("Error saving Video files:", e)

    print("Camera one completed")
    return df1

def run_function(func):
    return func()

if __name__ == '__main__':
    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=3) as pool:
        # Start the functions in parallel
        results = pool.map(run_function, [cameraOne])

    end_time = time.time()

    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Print results
    for i, result in enumerate(results, 1):
        # print(f"Result from Function {i}: {result}")
        if isinstance(result, pd.DataFrame):
            # If it is, append it to the dataframes list
            dataframes.append(result)
            print(f"DataFrame from Function {i} added to the array.")
        else:
            print(f"Result from Function {i} is not a DataFrame, skipped.")
     # Concatenate dataframes horizontally
    combined_df = pd.concat(dataframes, axis=1)
    try:
        combined_df.to_excel(os.path.join(os.path.dirname(__file__),'__22Telbow_angles.xlsx'), index=False)
        print("Excel files saved successfully.")
    except Exception as e:
        print("Error saving Excel files:", e)
    print(f"All functions completed in {end_time - start_time:.2f} seconds")