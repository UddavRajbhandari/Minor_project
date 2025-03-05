import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time

# Constants matching your training configuration
SEQUENCE_LENGTH = 45
# Expected feature shape: normalized landmarks (33*4=132) + velocity (132) + weighted confidence (33) = 297
LANDMARK_SHAPE = 33 * 4 * 2 + 33  # 297 features per frame
MOTION_THRESHOLD = 0.1  # Threshold to detect user activity

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5
)

@st.cache_resource
def load_model(model_path):

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

class RealTimeProcessor:
    def __init__(self, model):
        self.model = model
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.class_mapping = {
            0: 'barbell biceps curl',
            1: 'hammer curl',
            2: 'lat pulldown',
            3: 'lateral raise',
            4: 'pull Up',
            5: 'push-up',
            6: 'shoulder press'
        }
        
        self.LANDMARK_GROUPS = {
            'arms': [11, 12, 13, 14, 15, 16],
            'legs': [23, 24, 25, 26, 27, 28],
            'core': [23, 24, 11, 12],
            'upper_body': [11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        self.exercise_weights = {
            'barbell biceps curl': {'arms': 0.7, 'core': 0.2, 'legs': 0.1},
            'hammer curl': {'arms': 0.7, 'core': 0.2, 'legs': 0.1},
            'lat pulldown': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
            'lateral raise': {'arms': 0.6, 'core': 0.3, 'legs': 0.1},
            'pull Up': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
            'push-up': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
            'shoulder press': {'arms': 0.6, 'core': 0.3, 'legs': 0.1}
        }
        self.last_prediction = None  # will store an integer prediction index
        self.last_motion = 0.0
        
    def reset_buffer(self):
        self.frame_buffer.clear()
        
    def normalize_landmarks(self, landmarks):
        landmarks = np.array(landmarks).reshape(-1, 4)
        if landmarks.shape[0] != 33:
            return np.zeros(33 * 4)
        left_hip = landmarks[23][:3]
        right_hip = landmarks[24][:3]
        hip_center = (left_hip + right_hip) / 2
        landmarks[:, :3] -= hip_center
        return landmarks.flatten()
    
    def calculate_velocity(self, sequence):
        """Calculate velocity features for a sequence of normalized landmarks.
        The velocity is computed as the difference between consecutive frames.
        The output shape is the same as the input sequence.
        """
        
        velocity = np.zeros_like(sequence)
        velocity[1:] = sequence[1:] - sequence[:-1]
        return velocity
    
    def calculate_motion(self, sequence):
        """Calculate motion magnitude to detect user activity."""
        if len(sequence) < 2:
            return 0.0
        motion = np.mean(np.abs(sequence[-1] - sequence[-2]))
        return motion
    
    def calculate_weighted_confidence(self, landmarks, exercise_type):
        # Not used directly in prediction now; see get_prediction() below.
        if exercise_type not in self.exercise_weights:
            return np.zeros(33)
        weights = self.exercise_weights[exercise_type]
        weighted_scores = np.zeros(33)
        for group, weight in weights.items():
            group_landmarks = self.LANDMARK_GROUPS[group]
            for lm_idx in group_landmarks:
                weighted_scores[lm_idx] = landmarks[lm_idx].visibility * weight
        return weighted_scores
    
    def process_frame(self, frame):
        try:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] 
                     for lm in results.pose_landmarks.landmark]
                ).flatten()
                normalized = self.normalize_landmarks(landmarks)
            else:
                normalized = np.zeros(33 * 4)  # 132 values
            
            self.frame_buffer.append(normalized)
            self.last_motion = self.calculate_motion(self.frame_buffer)
            return True, frame
            
        except Exception as e:
            st.error(f"Frame processing error: {str(e)}")
            return False, frame

    def get_prediction(self, ignore_motion=False):
        if len(self.frame_buffer) != SEQUENCE_LENGTH:
            return None, 0.0
        try:
            # For video uploads, we might want to ignore motion threshold
            if not ignore_motion and self.last_motion < MOTION_THRESHOLD:
                return "No activity detected", 0.0

            sequence = np.array(self.frame_buffer)  # Shape: (45, 132)
            velocity = self.calculate_velocity(sequence)  # (45, 132)
            
            # Calculate weighted confidences for each frame in the sequence
            weighted_conf = []
            # Use a default exercise type if no prediction exists yet
            default_exercise = "push-up"
            if self.last_prediction is not None:
                exercise_type = self.class_mapping.get(self.last_prediction, default_exercise)
            else:
                exercise_type = default_exercise
            
            for frame in sequence:
                # Create dummy landmarks objects from the visibility values in the frame
                dummy_landmarks = [
                    type('Landmark', (object,), {'visibility': v})
                    for v in frame.reshape(-1, 4)[:, 3]
                ]
                frame_conf = np.zeros(33)
                weights = self.exercise_weights.get(exercise_type, {})
                for group, weight in weights.items():
                    group_indices = self.LANDMARK_GROUPS[group]
                    for idx in group_indices:
                        if idx < 33:
                            frame_conf[idx] = dummy_landmarks[idx].visibility * weight
                weighted_conf.append(frame_conf)
            
            weighted_conf = np.array(weighted_conf)  # Shape: (45, 33)
            
            # Final input assembly: concatenate the normalized landmarks, velocity, and weighted confidence features
            final_input = np.concatenate([sequence, velocity, weighted_conf], axis=1)  # Should be (45, 297)
            final_input = np.expand_dims(final_input, axis=0)  # (1, 45, 297)
            
            if final_input.shape != (1, SEQUENCE_LENGTH, LANDMARK_SHAPE):
                st.error(f"Invalid input shape: {final_input.shape}")
                return None, 0.0
                
            predictions = self.model.predict(final_input, verbose=0)
            self.last_prediction = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions))
            
            return self.class_mapping.get(self.last_prediction, "Unknown"), confidence
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0.0

    def generate_feedback(self, exercise):
        """Generate real-time feedback based on exercise type."""
        feedback_map = {
            'push-up': "Keep your back straight and lower your chest to the floor.",
            'barbell biceps curl': "Keep your elbows close to your body and avoid swinging.",
            'pull Up': "Pull your chin above the bar and lower yourself slowly.",
            'hammer curl': "Maintain controlled movement and avoid body swinging.",
            'lat pulldown': "Keep your torso upright and pull the bar to your chest.",
            'lateral raise': "Maintain slight elbow bend and control the weight.",
            'shoulder press': "Keep your core engaged and avoid arching your back."
        }
        return feedback_map.get(exercise, "Maintain proper form and control your movements.")

def main():
    st.title("Upper Body Fitness Coach")
    
    model_path = r"D:\minor\k_fold_CNN_LSTM_landmark\model_fold_3.keras"
    model = load_model(model_path)
    
    if not model:
        st.stop()
        
    processor = RealTimeProcessor(model)
    
    st.sidebar.header("Input Type")
    input_type = st.sidebar.radio("Select Source", ["Webcam", "Upload Video"])
    
    if input_type == "Webcam":
        run_webcam(processor)
    else:
        run_video_upload(processor)

def run_webcam(processor):
    st.header("Real-Time Webcam Analysis")
    
    # Use Streamlit session state to control webcam start/stop
    if 'camera_on' not in st.session_state:
         st.session_state.camera_on = False

    col1, col2 = st.columns(2)
    if col1.button("Start Camera"):
         st.session_state.camera_on = True
    if col2.button("Stop Camera"):
         st.session_state.camera_on = False

    if st.session_state.camera_on:
        processor.reset_buffer()
        cap = cv2.VideoCapture(0)
        pred_placeholder = st.empty()
        img_placeholder = st.empty()
        feedback_placeholder = st.empty()  
        last_pred_time = time.time()
        
        try:
            while cap.isOpened() and st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                success, processed_frame = processor.process_frame(frame)
                img_placeholder.image(processed_frame, channels="BGR")
                
                # Predict every 0.5 seconds (adjust if needed for faster feedback)
                if time.time() - last_pred_time > 0.5:
                    exercise, confidence = processor.get_prediction(ignore_motion=False)
                    if exercise and exercise != "No activity detected":
                        pred_placeholder.markdown(
                            f"**Exercise:** {exercise}<br>**Confidence:** {confidence:.1%}",
                            unsafe_allow_html=True
                        )
                        feedback = processor.generate_feedback(exercise)
                        feedback_placeholder.markdown(
                            f"**Feedback:** {feedback}",
                            unsafe_allow_html=True
                        )
                    last_pred_time = time.time()
                
                time.sleep(0.01)
        except Exception as e:
            st.error(f"Webcam error: {str(e)}")
        finally:
            cap.release()

def run_video_upload(processor):
    st.header("Video File Analysis")
    uploaded_file = st.file_uploader("Upload Exercise Video", type=["mp4", "mov"])
    
    if uploaded_file:
        processor.reset_buffer()
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_uploaded_video(temp_path, processor)

def process_uploaded_video(path, processor):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    pred_placeholder = st.empty()
    img_placeholder = st.empty()
    feedback_placeholder = st.empty()
    
    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processor.process_frame(frame)
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
            if frame_count % 5 == 0:
                img_placeholder.image(frame, channels="BGR")
                
            # For video uploads, ignore the motion threshold to ensure predictions are made
            if len(processor.frame_buffer) == SEQUENCE_LENGTH:
                exercise, confidence = processor.get_prediction(ignore_motion=True)
                if exercise and exercise != "No activity detected":
                    pred_placeholder.markdown(
                        f"**Detected:** {exercise}<br>**Confidence:** {confidence:.1%}",
                        unsafe_allow_html=True
                    )
                    feedback = processor.generate_feedback(exercise)
                    feedback_placeholder.markdown(
                        f"**Feedback:** {feedback}",
                        unsafe_allow_html=True
                    )
        # Optionally, clear the progress bar when done
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
    finally:
        cap.release()
        progress_bar.empty()

if __name__ == "__main__":
    main()
