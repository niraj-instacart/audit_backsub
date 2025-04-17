import streamlit as st
import cv2
import tempfile
import time
import hashlib
from backsub_processor import BackgroundSubtractorBuilder

# FPS of remote audit videos
FPS = 5

def layout():
    st.title("Remote Audit Background Sub Playground")

    video_path = upload_video()
    if video_path:
        # Overview section with built-in video player
        st.header("Video Overview")
        display_video(video_path)
        
        # Frame analysis section
        st.header("Frame Analysis Section")
        display_frame_analyzer(video_path)

def upload_video():
    """
    Handles video upload and returns the file path.
    """
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            return temp_file.name
    return None

def display_video(video_path):
    """
    Displays video with Streamlit's built-in video player.
    """
    st.video(video_path)

@st.cache_data
def load_video_frames(video_path):
    """
    Loads and caches all frames from the video.
    Returns a list of frames and the total frame count.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames, len(frames)

@st.cache_data
def analyze_frames(video_path, params_hash):
    """
    Process all frames with background subtraction and cache the results.
    
    Uses a hash of params instead of the full params object for better caching.
    """
    # Load frames 
    frames, _ = load_video_frames(video_path)
    
    # Retrieve actual parameters from session state
    backsub_params = st.session_state.current_backsub_params
    
    # Create background subtractor
    builder = BackgroundSubtractorBuilder()
    builder.with_method(backsub_params["method"])
    builder.with_history(backsub_params["history"])
    builder.with_shadow_detection(backsub_params["detect_shadows"])
    builder.with_var_threshold(backsub_params.get("var_threshold", 16))
    builder.with_learning_rate(backsub_params.get("learning_rate", -1))
    builder.with_dist2_threshold(backsub_params.get("dist2_threshold", 400))
    builder.with_morph_size(backsub_params["morph_size"])
    builder.with_min_contour_area(backsub_params["min_contour_area"])
    builder.with_threshold_value(backsub_params["threshold_value"])
    builder.with_motion_area_threshold(backsub_params["motion_area_threshold"])
    
    subtractor = builder.build()
    
    # Process each frame
    processed_frames = []
    motion_results = []
    
    # Create a placeholder for progress
    progress_ph = st.empty()
    
    for i, frame in enumerate(frames):
        # Update progress
        if i % 10 == 0:  # Update progress every 10 frames
            progress_ph.progress(i / len(frames))
        
        # Get foreground mask
        contours, mask = subtractor.get_foreground_objects(frame)
        has_motion, total_area = subtractor.has_significant_motion(frame)
        
        # Draw contours on a copy of the original frame
        contour_overlay = frame.copy()
        cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 2)
        
        # Store results
        processed_frames.append({
            'mask': mask,
            'contour_overlay': contour_overlay,
            'has_motion': has_motion,
            'total_area': total_area
        })
        motion_results.append(has_motion)
    
    # Clear progress
    progress_ph.empty()
    
    # Create a video file with contour overlay
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        output_path = temp_file.name
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    for result in processed_frames:
        # Convert to BGR for OpenCV
        bgr_frame = cv2.cvtColor(result['contour_overlay'], cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    
    return processed_frames, output_path, sum(motion_results)

def display_frame_analyzer(video_path):
    """
    Provides frame-by-frame analysis tools.
    """
    frames, total_frames = load_video_frames(video_path)
    
    # Get background subtraction parameters
    backsub_params = get_background_subtraction_params()
    
    # Store parameters in session state
    st.session_state.current_backsub_params = backsub_params
    
    # Create a stable hash of the parameters for caching
    params_str = str(sorted(backsub_params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Add button to process the entire video
    if st.button("Process Entire Video"):
        processed_frames, processed_video_path, motion_count = analyze_frames(
            video_path, params_hash
        )
        
        # Store in session state for frame analyzer to use
        st.session_state.processed_frames = processed_frames
        st.session_state.processed_video_path = processed_video_path
        st.session_state.has_processed = True
        
        # Show motion summary
        st.success(f"Video processed! Motion detected in {motion_count} of {total_frames} frames ({motion_count/total_frames*100:.1f}%)")
    
    # Display side-by-side videos if processed
    if 'has_processed' in st.session_state and st.session_state.has_processed:
        st.subheader("Processed Video")
        
        # Display side-by-side videos
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Original Video")
            st.video(video_path)
        
        with col2:
            st.caption("Background Subtraction (Contour Overlay)")
            st.video(st.session_state.processed_video_path)
    
    # Frame-by-frame analyzer
    st.subheader("Frame Analysis")
    
    # Add toggle for frames/seconds, default to seconds
    use_frames = st.checkbox("Show frame numbers instead of seconds")
    
    if use_frames:
        # Frame selection slider
        frame_idx = st.slider("Select Frame", 0, max(0, total_frames - 1), 0)
        st.caption(f"Time: {frame_idx/FPS:.1f}s")
    else:
        duration = total_frames / FPS
        current_time = st.slider("Select Time (seconds)", 0.0, duration, 0.0, step=0.2)
        frame_idx = min(int(current_time * FPS), total_frames - 1)
        st.caption(f"Frame {frame_idx} at {current_time:.1f}s")
    
    # Display frames side by side
    col1, col2 = st.columns(2)
    
    # Display original frame in the first column
    with col1:
        st.subheader("Original Frame")
        st.image(frames[frame_idx], caption=f"Frame {frame_idx}")
    
    # If processed, display the processed frame
    if 'has_processed' in st.session_state and st.session_state.has_processed:
        with col2:
            # Get the processed frame data
            processed_frame = st.session_state.processed_frames[frame_idx]
            
            # Add radio to select what to display
            display_option = st.radio(
                "Display",
                ["Foreground Mask", "Contour Overlay"],
                horizontal=True
            )
            
            if display_option == "Foreground Mask":
                st.image(processed_frame['mask'], caption=f"Foreground Mask (Frame {frame_idx})")
            else:
                st.image(processed_frame['contour_overlay'], caption=f"Contour Overlay (Frame {frame_idx})")
            
            # Show motion status
            if processed_frame['has_motion']:
                st.success(f"✅ Motion detected! Area: {processed_frame['total_area']:.1f} px²")
            else:
                st.warning(f"❌ No significant motion. Area: {processed_frame['total_area']:.1f} px²")
    else:
        with col2:
            st.info("Process the entire video to see background subtraction results")

def get_background_subtraction_params():
    """
    Creates UI elements for background subtraction parameters and returns their values.
    """
    with st.expander("Background Subtraction Parameters"):
        # Method selection
        method = st.selectbox(
            "Background Subtraction Method",
            ["MOG2", "KNN", "GMG"],
            help="Select the background subtraction algorithm to use"
        )
        
        # Common parameters
        history = st.slider(
            "History Length",
            min_value=1,
            max_value=1000,
            value=500,
            help="Number of frames used to build the background model"
        )
        
        detect_shadows = st.checkbox(
            "Detect Shadows",
            value=True,
            help="Enable shadow detection"
        )
        
        # Method-specific parameters
        if method == "MOG2":
            var_threshold = st.slider(
                "Variance Threshold",
                min_value=1,
                max_value=100,
                value=16,
                help="Threshold on the squared Mahalanobis distance to decide if pixel is background"
            )
            
            learning_rate = st.slider(
                "Learning Rate",
                min_value=-1.0,
                max_value=1.0,
                value=-1.0,
                help="Learning rate for background model. -1 for auto, 0-1 for manual"
            )
        else:
            var_threshold = 16
            learning_rate = -1
            
        if method == "KNN":
            dist2_threshold = st.slider(
                "Distance Threshold",
                min_value=1,
                max_value=1000,
                value=400,
                help="Threshold on the squared distance to decide if pixel is background"
            )
        else:
            dist2_threshold = 400
        
        # Post-processing parameters
        st.subheader("Post-processing")
        
        morph_size = st.slider(
            "Morphological Operation Size",
            min_value=0,
            max_value=21,
            value=5,
            step=2,  # Only odd values for kernel size
            help="Size of kernel for noise removal (0 to disable)"
        )
        
        min_contour_area = st.slider(
            "Minimum Object Size (px²)",
            min_value=0,
            max_value=5000,
            value=500,
            help="Filter out objects smaller than this area"
        )
        
        threshold_value = st.slider(
            "Threshold Value",
            min_value=0,
            max_value=255,
            value=127,
            help="Threshold for converting to binary mask"
        )
        
        # Motion detection parameters
        st.subheader("Motion Detection")
        
        motion_area_threshold = st.slider(
            "Motion Area Threshold (px²)",
            min_value=0,
            max_value=50000,
            value=5000,
            help="Minimum total area to classify as significant motion"
        )
        
        # Return parameters as a dictionary
        return {
            "method": method,
            "history": history,
            "detect_shadows": detect_shadows,
            "var_threshold": var_threshold,
            "learning_rate": learning_rate,
            "dist2_threshold": dist2_threshold,
            "morph_size": morph_size,
            "min_contour_area": min_contour_area,
            "threshold_value": threshold_value,
            "motion_area_threshold": motion_area_threshold
        }

# Run the app
layout()

