import cv2
import numpy as np 
import easygui
from PyQt5.QtCore import QObject

class OpticalFlowAlgorithm:
   
    def __init__(self, video_dir : str, output_dir: str, save_frames = False):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.cap = None 
        self.previous_frame = None
        self.resize_shape = (128, 128)
        self.save_frames = save_frames

    def set_video_dir(self, video_dir):
        self.video_dir = video_dir
    
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def _open_video_capture(self,):
        self.cap = cv2.VideoCapture(self.video_dir)
    
    def _process_frame(self, frame, frame_idx,  first_frame = False):
        frame = cv2.resize(frame, self.resize_shape)
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame, np.float32) # create HSV image for optical flow 
        hsv[..., 1] = 1.0 # set saturation to 1

        if not first_frame:
            # # calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(self.previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # # Calculate Normal TVL1 optical flow 
            # tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            # flow = tvl1.calc(self.previous_frame, current_frame, None)
            # convert to polar coordinates get magnitude and angle
            magnitude, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1], angleInDegrees=True
            )
            # set hue according to the angle of optical flow
            hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))
            
            # set value according to the normalized magnitude of optical flow
            hsv[..., 2] = cv2.normalize(
                magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
            )
            # multiply each pixel value to 255
            hsv_8u = np.uint8(hsv * 255.0)

            # convert hsv to bgr
            bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)
            # save the frame
            if self.save_frames:
                cv2.imwrite(f"{self.output_dir}/{frame_idx}.png", bgr)


        self.previous_frame = current_frame


    def run(self):
        self._open_video_capture()
        curr_frame_idx = 0

        # read the first frame
        ret, frame = self.cap.read()
        # proceed if frame reading was successful
        if ret:
            self._process_frame(frame, curr_frame_idx, first_frame=True)

            # loop over all the frames
            while True:

                # capture frame-by-frame
                ret, frame = self.cap.read()

                # Termination condition, when No frame is left to read, EXIT
                if not ret:
                    break

                self._process_frame(frame, curr_frame_idx)

                # increase the frame index
                curr_frame_idx +=1
             
        self.cap.release()

        

class OpticalFlowGUI(QObject):
    def __init__(self, algorithm: OpticalFlowAlgorithm):
        super().__init__()
        self.algorithm = algorithm
    
    def set_video_dir(self, video_dir):
        self.algorithm.video_dir = video_dir
    
    def set_output_dir(self, output_dir):
        self.algorithm.output_dir = output_dir

    def run(self):
        self.algorithm.run()
    

def run_optical_flow_algorithm_example():
    
    video_dir = easygui.fileopenbox("Choose a video file", "Video File", filetypes=["*.mp4", "*.avi"])
    output_dir = "temp/"

    optical_flow = OpticalFlowAlgorithm(video_dir, output_dir, save_frames=True)
    optical_flow.run()


def run_optical_flow_gui_example():
    
    video_dir = easygui.fileopenbox("Choose a video file", "Video File", filetypes=["*.mp4", "*.avi"])
    output_dir = "temp/"

    algorithm = OpticalFlowAlgorithm(video_dir, output_dir, save_frames=True)
    
    gui = OpticalFlowGUI(algorithm)
    gui.run()

if __name__ == "__main__":
    # run_optical_flow_algorithm_example()
    run_optical_flow_gui_example()