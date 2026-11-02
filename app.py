import streamlit as st
from ultralytics import YOLO
import os
import cv2

# Set the page to be a bit wider to fit our camera and sidebar nicely
st.set_page_config(page_title="ADAS Vision App", layout="wide")

# ==============================
# UI CONTROLS (Sidebar)
# ==============================
st.sidebar.title("⚙️ AI Settings")
st.sidebar.write("Adjust how strict the AI is when detecting objects.")

# 1. Advanced Slider Control
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.3, 
    step=0.05,
    help="Higher values mean the AI will only show objects it is very sure about."
)

st.title("🚗 Real-Time ADAS Object Detector")
st.write("Upload a dashcam video or use your live webcam to detect objects.")

# ==============================
# LOAD AI MODEL
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================
# APP MODES (Radio Buttons)
# ==============================
app_mode = st.radio("Choose Input Mode:", ["📁 Video Upload", "📷 Live Webcam"])

if app_mode == "📁 Video Upload":
    uploaded_file = st.file_uploader("Upload a driving video", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.success("Video ready! Click the button below.")
        
        if st.button("Start AI Processing"):
            with st.spinner(f"AI is analyzing the frames at {int(conf_threshold*100)}% confidence..."):
                input_path = "uploaded_video.mp4"
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Run YOLO with the user's slider setting!
                results = model.predict(
                    source=input_path, 
                    conf=conf_threshold, # Connecting the slider to the AI
                    save=True, 
                    project="output", 
                    name="webapp_run", 
                    exist_ok=True
                )
                
                save_dir = results[0].save_dir
                output_file_path = None
                
                if os.path.exists(save_dir):
                    for file in os.listdir(save_dir):
                        if file.endswith(".mp4") or file.endswith(".avi"):
                            output_file_path = os.path.join(save_dir, file)
                            break
                
                if output_file_path and os.path.exists(output_file_path):
                    st.success("✅ Processing Complete!")
                    with open(output_file_path, "rb") as processed_video:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=processed_video,
                            file_name="AI_detected_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Error: Could not locate the output video.")

elif app_mode == "📷 Live Webcam":
    st.write("Click 'Start' to turn on your Mac's camera. Try adjusting the slider on the left while it runs!")
    
    # Use session state to handle the Start/Stop loop safely
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)
    
    if col1.button("🟢 Start Webcam"):
        st.session_state.run_webcam = True
    if col2.button("🔴 Stop Webcam"):
        st.session_state.run_webcam = False

    # This creates an empty box on the screen where we will blast the video frames
    frame_placeholder = st.empty()
    
    if st.session_state.run_webcam:
        # Open the default Mac webcam (0)
        cap = cv2.VideoCapture(0)
        
        while st.session_state.run_webcam:
            success, frame = cap.read()
            if not success:
                st.error("Failed to access the webcam. Please ensure VS Code/Terminal has camera permissions.")
                break
                
            # Run YOLO on the live frame using the slider threshold
            results = model(frame, conf=conf_threshold)
            
            # Draw the bounding boxes on the frame
            annotated_frame = results[0].plot()
            
            # Streamlit requires RGB colors, but OpenCV uses BGR. We must convert it!
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Send the image to the web app
            frame_placeholder.image(annotated_frame, channels="RGB")
            
        cap.release()
