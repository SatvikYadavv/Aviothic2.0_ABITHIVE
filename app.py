import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import base64
import numpy as np
from PIL import Image

model = YOLO("best.pt")

st.set_page_config(page_title="🐄 PashuScan", layout="centered",
                   initial_sidebar_state="auto")

# Updated CSS with dark theme and high contrast
st.markdown(
    """
    <style>
        /* Base Styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body, .stApp {
            background: #0a0a0a;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
            min-height: 100vh;
        }

        /* Main Container */
        .main-container {
            background: rgba(25, 25, 35, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 1200px;
        }

        /* Header with Patriotic Gradient */
        .patriotic-header {
            background: linear-gradient(90deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-weight: 800;
            font-size: clamp(2.5rem, 6vw, 4rem);
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .subtitle {
            text-align: center;
            color: #e0e0e0;
            font-size: clamp(1rem, 2.5vw, 1.3rem);
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Text Elements - High Contrast */
        .stMarkdown, .stText, .stTitle, .stHeader {
            color: #ffffff !important;
        }

        /* Radio Buttons - Dark Theme */
        .stRadio > div {
            background: rgba(40, 40, 50, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px solid #444;
            transition: all 0.3s ease;
        }

        .stRadio > div:hover {
            border-color: #FF9933;
            box-shadow: 0 4px 15px rgba(255, 153, 51, 0.3);
        }

        .stRadio label {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #ffffff !important;
        }

        /* Buttons - Patriotic Colors */
        .stButton button {
            background: linear-gradient(135deg, #FF9933 0%, #138808 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 153, 51, 0.4);
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 153, 51, 0.6);
            color: white;
        }

        /* File Uploader Styling */
        .stFileUploader > div {
            border: 2px dashed #FF9933;
            border-radius: 15px;
            padding: 2rem;
            background: rgba(255, 153, 51, 0.1);
            transition: all 0.3s ease;
        }

        .stFileUploader > div:hover {
            border-color: #138808;
            background: rgba(19, 136, 8, 0.1);
        }

        /* Laser Animation */
        .laser-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            margin: 2rem 0;
            border: 2px solid #333;
        }

        .laser {
            position: absolute;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #FF4500 20%, 
                #FF9933 50%, 
                #138808 80%, 
                transparent 100%);
            pointer-events: none;
            mix-blend-mode: screen;
            top: 0;
            left: 0;
            animation: scan 2s ease-in-out infinite;
            z-index: 10;
            box-shadow: 0 0 20px #FF4500;
        }

        .laser.paused {
            animation-play-state: paused;
            opacity: 0;
        }

        @keyframes scan {
            0% { top: 0; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        /* Prediction Results */
        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .prediction-text {
            font-size: 1.3rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1rem;
            color: #ffffff;
        }

        /* Info and Warning Boxes */
        .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 10px;
            border-left: 4px solid;
            color: #ffffff !important;
        }

        .stInfo {
            background: rgba(41, 128, 185, 0.2) !important;
            border-left-color: #2980b9 !important;
        }

        .stSuccess {
            background: rgba(39, 174, 96, 0.2) !important;
            border-left-color: #27ae60 !important;
        }

        .stWarning {
            background: rgba(243, 156, 18, 0.2) !important;
            border-left-color: #f39c12 !important;
        }

        .stError {
            background: rgba(231, 76, 60, 0.2) !important;
            border-left-color: #e74c3c !important;
        }

        /* Text Input */
        .stTextInput input {
            background: rgba(50, 50, 60, 0.9) !important;
            color: #ffffff !important;
            border: 2px solid #555 !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }

        .stTextInput input:focus {
            border-color: #FF9933 !important;
            box-shadow: 0 0 10px rgba(255, 153, 51, 0.3) !important;
        }

        /* Patriotic Badge */
        .patriotic-badge {
            background: linear-gradient(90deg, #FF9933, #FFFFFF, #138808);
            color: #2c3e50;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        /* Camera Container */
        .camera-box {
            background: rgba(40, 40, 50, 0.9);
            border-radius: 15px;
            padding: 1.5rem;
            border: 2px solid #444;
            margin: 1rem 0;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-container {
                padding: 1rem;
                margin: 0.5rem;
            }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1a1a2e;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#FF9933, #138808);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(#e6821d, #0f7206);
        }

        /* Streamlit Native Element Overrides */
        .st-bd, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
            color: #ffffff !important;
        }

        /* Ensure all text is visible */
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #ffffff !important;
        }

        /* Specific streamlit class overrides */
        .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Main App Content
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header with Patriotic Touch
st.markdown('<h1 class="patriotic-header">🐄 PashuScan</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Cattle Breed Recognition • Made in India 🇮🇳</p>', unsafe_allow_html=True)

st.markdown('<div class="patriotic-badge">Empowering Indian Farmers with AI Technology</div>', unsafe_allow_html=True)

# Option Selection
st.markdown("### 📷 Choose Input Method")
option = st.radio("", ["Image Upload", "Real-time Camera"], 
                  help="Select how you want to analyze cattle images")

feedback_file = "feedback.csv"

def save_feedback(image_path, predicted, correct=None):
    new_row = {"image_path": image_path, "predicted": predicted, "correct": correct}

    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(feedback_file, index=False)

def show_image_with_laser(image_bgr, scanning=True):
    _, im_jpg = cv2.imencode('.jpg', image_bgr)
    im_bytes = im_jpg.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode()

    laser_class = "laser" if scanning else "laser paused"

    st.markdown(
        f"""
        <div class="laser-container">
            <img src="data:image/jpg;base64,{im_b64}" style="width:100%; display:block;">
            <div class="{laser_class}"></div>
        </div>
        """, unsafe_allow_html=True
    )

def show_camera_with_laser(image_bgr, scanning=True):
    _, im_jpg = cv2.imencode('.jpg', image_bgr)
    im_bytes = im_jpg.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode()

    laser_class = "laser" if scanning else "laser paused"

    st.markdown(
        f"""
        <div class="laser-container">
            <img src="data:image/jpg;base64,{im_b64}" style="width:100%; display:block;">
            <div class="{laser_class}"></div>
        </div>
        """, unsafe_allow_html=True
    )

if option == "Image Upload":
    st.markdown("### 📁 Upload Cattle Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                   help="Upload an image of cattle for breed identification")
    
    if uploaded_file:
        temp_file_path = "temp.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Show scanning laser while prediction is running
        img = cv2.imread(temp_file_path)
        show_image_with_laser(img, scanning=True)
        st.info("🔍 Scanning image with AI... Please wait.")

        results = model(temp_file_path, conf=0.4, save=False)

        has_prediction = False
        for r in results:
            if len(r.boxes) > 0:
                has_prediction = True
                im_bgr = r.plot()
                # Stop laser scan animation after prediction
                show_image_with_laser(im_bgr, scanning=False)

                predicted_classes = [r.names[int(c)] for c in r.boxes.cls]
                
                # Display prediction in a styled card
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-text">
                            🎯 Detected Breed: {", ".join(predicted_classes)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                # Feedback buttons
                st.markdown("### 💬 Help Improve Our AI")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Prediction Correct", use_container_width=True):
                        save_feedback(temp_file_path, predicted_classes, correct="Same")
                        st.success("🙏 Thank you for your feedback!")
                with col2:
                    if st.button("❌ Prediction Wrong", use_container_width=True):
                        correct_label = st.text_input("Please enter the correct breed name:")
                        if correct_label:
                            save_feedback(temp_file_path, predicted_classes, correct_label)
                            st.success(f"📝 Feedback recorded! Correct breed: {correct_label}")

        if not has_prediction:
            st.warning("⚠️ No cattle breed detected. Please try with a clearer image.")

elif option == "Real-time Camera":
    st.markdown("### 📸 Live Camera Detection")
    st.info("Real-time cattle breed detection using your camera")
    
    # Initialize session state for camera control
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Camera control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.camera_active:
            if st.button("🎥 Start Camera", use_container_width=True):
                st.session_state.camera_active = True
                st.rerun()
    
    with col2:
        if st.session_state.camera_active:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()
    
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access camera. Please check permissions.")
        else:
            # Create placeholders
            camera_placeholder = st.empty()
            info_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            st.success("🔍 Camera active - Point at cattle for breed detection")
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Perform detection
                results = model(frame, conf=0.4, verbose=False)
                
                has_prediction = False
                predicted_classes = []
                
                for r in results:
                    if len(r.boxes) > 0:
                        has_prediction = True
                        frame = r.plot()
                        predicted_classes = [r.names[int(c)] for c in r.boxes.cls]
                
                # Convert for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Show camera feed with laser
                show_camera_with_laser(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), 
                                     scanning=not has_prediction)
                
                if has_prediction:
                    info_placeholder.success(
                        f"**🎯 Detected Breed:** {', '.join(predicted_classes)}"
                    )
                    
                    # Feedback section
                    with feedback_placeholder.container():
                        st.markdown("### 💬 Is this correct?")
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            if st.button("✅ Yes, Correct", key="cam_correct"):
                                temp_path = "camera_feedback.jpg"
                                cv2.imwrite(temp_path, frame)
                                save_feedback(temp_path, predicted_classes, "Same")
                                st.success("Feedback saved! 👍")
                        
                        with feedback_col2:
                            if st.button("❌ No, Wrong", key="cam_wrong"):
                                correct_label = st.text_input("Correct breed name:")
                                if correct_label:
                                    temp_path = "camera_feedback.jpg"
                                    cv2.imwrite(temp_path, frame)
                                    save_feedback(temp_path, predicted_classes, correct_label)
                                    st.success(f"Updated to: {correct_label}")
                else:
                    info_placeholder.info("📡 Adjust camera - Looking for cattle...")
                    feedback_placeholder.empty()
                
                cv2.waitKey(1)
            
            cap.release()
            st.info("📹 Camera session ended")
    
    else:
        st.info("👆 Click 'Start Camera' to begin live detection")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #e0e0e0; padding: 1rem;">
        <p>🚀 <strong style="color: #ffffff;">PashuScan</strong> - Revolutionizing Cattle Management with AI</p>
        <p>🇮🇳 Proudly Indian | 🤝 Supporting Farmers | 🔬 Powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True
)