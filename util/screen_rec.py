import cv2
import numpy as np
import pyautogui
import threading
import time


class ScreenRecorder:
    def __init__(self, output_file_dir=f"./outputs/", max_duration = 1000,fps = 30):
        self.output_file_dir = output_file_dir
        self.max_duration = max_duration
        self.recording = False
        self.start_time = 0
        self.thread = None
        self.fps = fps
        

    def start_rec(self):
        self.recording = True
        self.start_time = time.time()
        # Create and start the recording thread as a daemon thread
        self.thread = threading.Thread(target=self._record_frames(),daemon=True)
        self.thread.start()

    def end_rec(self):
        self.recording = False
        if self.thread is not None:
            self.thread.join()

    def _record_screen(self):
        # Get screen size
        screen_size = (1920, 1080)  # Update with your screen resolution

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_file = cv2.VideoWriter(self.output_file_dir, fourcc, 30.0, screen_size)

        # Set start time
        start_time = time.time()

        # Record screen until max duration is reached
        while self.recording and time.time() - start_time < self.max_duration:
            
            time.sleep(60/self.fps)
            # Capture screen frame
            frame = pyautogui.screenshot()
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Write frame to video file
            output_file.write(frame)

        # Release the video writer and destroy any open windows
        output_file.release()
        cv2.destroyAllWindows()
    
    def _record_frames(self):
        
        frame_width = 1920
        frame_height = 1080
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        output_file = f"{self.output_file_dir}/video_{timestamp}.avi"
        out = cv2.VideoWriter(output_file, fourcc, self.fps, (frame_width, frame_height))
        while self.recording and (time.time() - self.start_time) < self.max_duration:
            
            frame = pyautogui.screenshot()
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        
        # Release the VideoWriter and close the video file
        out.release()