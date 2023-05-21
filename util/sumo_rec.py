import cv2
import numpy as np
import threading
import time
import traci
import os

class SumoRecorder:
    def __init__(self, output_file_dir=f"./frames", max_duration = 1e3):
        self.output_file_dir = output_file_dir
        self.max_duration = max_duration
        self.recording = False
        self.start_time = 0


    def _start_rec(self):
        self.recording = True
        self.start_time = time.time()
        # Create and start the recording thread as a daemon thread

    def _end_rec(self):
        self.recording = False

    
    def _record_frame(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        traci.gui.screenshot(traci.gui.DEFAULT_VIEW,filename=f"{self.output_file_dir}/f_{timestamp}.png")


