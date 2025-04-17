import cv2
import numpy as np

class BackgroundSubtractor:
    """
    Background subtraction processor using OpenCV's background subtraction algorithms.
    """
    
    def __init__(self, method, history=500, detect_shadows=True, var_threshold=16, 
                 learning_rate=-1, dist2_threshold=400, morph_size=5, 
                 min_contour_area=500, threshold_value=127, motion_area_threshold=5000):
        """
        Initialize the background subtractor with parameters.
        
        Args:
            method: Background subtraction method ("MOG2", "KNN", or "GMG")
            history: Length of history (frames)
            detect_shadows: Whether to detect shadows
            var_threshold: Variance threshold for MOG2
            learning_rate: Learning rate for MOG2
            dist2_threshold: Distance threshold for KNN
            morph_size: Size of morphological operation kernel
            min_contour_area: Minimum contour area to filter small noise
            threshold_value: Threshold value for binary mask
            motion_area_threshold: Minimum total contour area to classify as significant motion
        """
        self.method_name = method
        
        # Create the appropriate OpenCV background subtractor
        if method == "MOG2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        elif method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2_threshold,
                detectShadows=detect_shadows
            )
        elif method == "GMG":
            self.subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
                initializationFrames=history
            )
        else:
            raise ValueError(f"Unsupported background subtraction method: {method}")
        
        # Post-processing parameters
        self.morph_size = morph_size
        self.min_contour_area = min_contour_area
        self.threshold_value = threshold_value
        
        # Add motion area threshold
        self.motion_area_threshold = motion_area_threshold
        
    def apply(self, frame):
        """
        Apply background subtraction to a frame.
        
        Args:
            frame: The input frame (RGB or BGR format)
            
        Returns:
            mask: The foreground mask
        """
        # Make sure frame is in BGR format for OpenCV
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Convert from RGB to BGR if needed
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Frame must be a color image with 3 channels")
        
        # Apply background subtraction
        mask = self.subtractor.apply(bgr_frame)
        
        # Post-processing options
        if self.morph_size > 0:
            # Create kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size))
            # Apply morphological operations to remove noise and fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def get_method_name(self):
        """Returns the name of the background subtraction method."""
        return self.method_name
    
    def get_foreground_objects(self, frame):
        """
        Extract foreground objects from the frame.
        
        Returns:
            contours: List of contours for foreground objects
            mask: The processed foreground mask
        """
        mask = self.apply(frame)
        
        # Threshold the mask if needed
        _, thresh = cv2.threshold(mask, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_contour_area]
        
        return filtered_contours, mask
    
    def has_significant_motion(self, frame):
        """
        Determines if the frame contains significant motion based on the total area of foreground objects.
        
        Args:
            frame: The input frame
            
        Returns:
            bool: True if significant motion is detected, False otherwise
            float: The total contour area
        """
        contours, _ = self.get_foreground_objects(frame)
        
        # Calculate total area of all contours
        total_area = sum(cv2.contourArea(cnt) for cnt in contours)
        
        # Check if total area exceeds the threshold
        has_motion = total_area >= self.motion_area_threshold
        
        return has_motion, total_area


class BackgroundSubtractorBuilder:
    """
    Builder for creating background subtractors with various configurations.
    """
    
    def __init__(self):
        # Default parameters
        self.method = "MOG2"
        self.history = 500
        self.detect_shadows = True
        
        # Method-specific parameters
        # MOG2
        self.var_threshold = 16
        self.learning_rate = -1  # Auto learning rate
        
        # KNN
        self.dist2_threshold = 400
        
        # Add post-processing parameters
        self.morph_size = 5
        self.min_contour_area = 500
        self.threshold_value = 127
        
        # Add motion detection threshold
        self.motion_area_threshold = 5000
        
    def with_method(self, method):
        """Set the background subtraction method."""
        self.method = method
        return self
        
    def with_history(self, history):
        """Set history length."""
        self.history = history
        return self
        
    def with_shadow_detection(self, detect_shadows):
        """Enable or disable shadow detection."""
        self.detect_shadows = detect_shadows
        return self
        
    def with_var_threshold(self, var_threshold):
        """Set variance threshold (MOG2 only)."""
        self.var_threshold = var_threshold
        return self
        
    def with_learning_rate(self, learning_rate):
        """Set learning rate (MOG2 only)."""
        self.learning_rate = learning_rate
        return self
        
    def with_dist2_threshold(self, dist2_threshold):
        """Set distance threshold (KNN only)."""
        self.dist2_threshold = dist2_threshold
        return self
    
    def with_morph_size(self, size):
        """Set morphological operation kernel size for noise removal."""
        self.morph_size = size
        return self
    
    def with_min_contour_area(self, area):
        """Set minimum contour area to filter small noise."""
        self.min_contour_area = area
        return self
    
    def with_threshold_value(self, value):
        """Set threshold value for binary mask."""
        self.threshold_value = value
        return self
    
    def with_motion_area_threshold(self, area):
        """Set threshold for significant motion detection."""
        self.motion_area_threshold = area
        return self
    
    def build(self):
        """
        Build and return a configured BackgroundSubtractor instance.
        """
        return BackgroundSubtractor(
            method=self.method,
            history=self.history,
            detect_shadows=self.detect_shadows,
            var_threshold=self.var_threshold,
            learning_rate=self.learning_rate,
            dist2_threshold=self.dist2_threshold,
            morph_size=self.morph_size,
            min_contour_area=self.min_contour_area,
            threshold_value=self.threshold_value,
            motion_area_threshold=self.motion_area_threshold
        )


# Example usage:
# subtractor = BackgroundSubtractorBuilder().with_method("MOG2").with_history(200).build()
# or
# params = {"method": "MOG2", "history": 200, "detect_shadows": True}
# subtractor = BackgroundSubtractorBuilder().build_from_params(params) 
