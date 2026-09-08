import cv2
import numpy as np

class OpticalFlowEstimator:
    def __init__(self):
        """
        Initializes the Dense Optical Flow Estimator.
        """
        self.prev_gray = None

    def estimate(self, frame):
        """
        Calculates dense optical flow between the previous frame and the current frame.
        Uses Farneback's algorithm.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # grayscale cevirne hiz olmasi icin 
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            # Return zero motion map
            return np.zeros_like(curr_gray, dtype=np.float32)

        # Calculate dense optical flow
        # parameters: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        #vektorleri hiza cevirme
        # flow is a 2-channel array (dx, dy). Compute magnitude (speed) of motion
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        self.prev_gray = curr_gray
        return magnitude
