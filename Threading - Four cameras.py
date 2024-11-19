import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import threading
import time
import os
import queue

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def check_cameras():
    """Check if all four cameras are available."""
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(2)
    cap4 = cv2.VideoCapture(3)
    
    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened() or not cap4.isOpened():
        if cap1.isOpened():
            cap1.release()
        if cap2.isOpened():
            cap2.release()
        if cap3.isOpened():
            cap3.release()
        if cap4.isOpened():
            cap4.release()
        raise RuntimeError("Could not open all four cameras")
        
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()

class PoseDetector:
    def __init__(self, camera_id, output_prefix, start_event):
        self.camera_id = camera_id
        self.output_prefix = output_prefix
        self.start_event = start_event
        self.counter = 0
        self.stage = None
        self.angle_values = []
        self.running = True
        
        # Video writer settings
        self.size = (640, 480)
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create output directory
        self.output_dir = f"frames_{camera_id}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def get_landmarks(self):
        return {
            'up': mp_pose.PoseLandmark.LEFT_HIP,
            'middle': mp_pose.PoseLandmark.LEFT_KNEE,
            'low': mp_pose.PoseLandmark.LEFT_ANKLE
        }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_camera(self):
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Configure camera
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        
        # Initialize video writer
        out = cv2.VideoWriter(
            os.path.join(os.path.dirname(__file__), f'{self.output_prefix}_output.mp4'),
            self.fourcc, self.fps, self.size
        )

        # Wait for start signal
        print(f"Camera {self.camera_id} ready and waiting...")
        self.start_event.wait()
        
        start_time = time.time()
        frame_number = 0
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    marks = self.get_landmarks()
                    
                    # Get coordinates
                    up_point = [landmarks[marks['up'].value].x, landmarks[marks['up'].value].y]
                    middle_point = [landmarks[marks['middle'].value].x, landmarks[marks['middle'].value].y]
                    low_point = [landmarks[marks['low'].value].x, landmarks[marks['low'].value].y]
                    
                    # Calculate angle
                    angle = self.calculate_angle(up_point, middle_point, low_point)
                    
                    # Visualize angle
                    cv2.putText(image, f"{angle:.1f}",
                              tuple(np.multiply(middle_point, [640, 480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA)
                    
                    # Track exercise state
                    if angle > 169:
                        self.stage = "UP"
                    if angle <= 90 and self.stage == 'UP':
                        self.stage = "DOWN"
                        self.counter += 1
                    
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Save data
                    self.angle_values.append({
                        'Time': time.time() - start_time,
                        'Angle': angle,
                        'Camera': self.camera_id
                    })
                    
                    # Save frame
                    frame_filename = os.path.join(self.output_dir, f"frame_{frame_number}.jpg")
                    cv2.imwrite(frame_filename, image)
                    
                except Exception as e:
                    print(f"Error in camera {self.camera_id}:", e)
                
                # Write frame to video
                out.write(image)
                
                # Display frame
                cv2.imshow(f'Camera {self.camera_id}', image)
                
                frame_number += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                
                # Control frame rate
                time.sleep(0.03)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyWindow(f'Camera {self.camera_id}')
        
        # Save data to Excel
        df = pd.DataFrame(self.angle_values)
        df.to_excel(os.path.join(os.path.dirname(__file__), 
                                f'{self.output_prefix}_angles_Threading.xlsx'), 
                   index=False)
        
        return df

def main():
    try:
        # Check cameras before starting
        check_cameras()
        
        # Create synchronization event
        start_event = threading.Event()
        
        # Initialize detectors
        detector1 = PoseDetector(camera_id=0, output_prefix='cam1', start_event=start_event)
        detector2 = PoseDetector(camera_id=1, output_prefix='cam2', start_event=start_event)
        detector3 = PoseDetector(camera_id=2, output_prefix='cam3', start_event=start_event)
        detector4 = PoseDetector(camera_id=3, output_prefix='cam4', start_event=start_event)
        
        # Create threads
        thread1 = threading.Thread(target=detector1.process_camera)
        thread2 = threading.Thread(target=detector2.process_camera)
        thread3 = threading.Thread(target=detector3.process_camera)
        thread4 = threading.Thread(target=detector4.process_camera)
        
        # Start threads
        print("Starting camera threads...")
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        
        # Small delay to ensure all cameras are ready
        time.sleep(2)
        
        # Signal all cameras to start processing
        print("Starting processing...")
        start_event.set()
        
        # Wait for threads to complete
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
        
        # Combine results
        df1 = pd.DataFrame(detector1.angle_values)
        df2 = pd.DataFrame(detector2.angle_values)
        df3 = pd.DataFrame(detector3.angle_values)
        df4 = pd.DataFrame(detector4.angle_values)
        
        # Merge dataframes
        combined_df = pd.merge_asof(
            pd.merge_asof(
                pd.merge_asof(
                    df1, df2, on='Time', suffixes=('_cam1', '_cam2'), 
                    direction='nearest'
                ),
                df3, on='Time', suffixes=('', '_cam3'), 
                direction='nearest'
            ),
            df4, on='Time', suffixes=('', '_cam4'), 
            direction='nearest'
        )
        
        # Save combined results
        combined_df.to_excel(os.path.join(os.path.dirname(__file__), 
                                        'combined_angles_Threading.xlsx'),
                           index=False)
        
        print("Processing completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()