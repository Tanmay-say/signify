import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import mediapipe as mp

# Set page config for mobile-friendly layout
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 3em;
            margin: 1em 0;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

# Initialize MediaPipe Hands
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    return mp_hands, mp_drawing, mp_drawing_styles, hands

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

def process_frame(frame, hands, mp_hands, mp_drawing, mp_drawing_styles):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
    
    return frame, data_aux, x_, y_

def main():
    st.title("✋ Sign Language Recognition")
    st.markdown("""
        This app recognizes American Sign Language (ASL) gestures in real-time using your device's camera.
        Show your hand to the camera to see the recognized sign.
    """)
    
    # Initialize model and MediaPipe
    model = load_model()
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera. Please check your camera connection.")
                break
                
            # Process the frame
            processed_frame, data_aux, x_, y_ = process_frame(frame, hands, mp_hands, mp_drawing, mp_drawing_styles)
            
            if data_aux:  # If hand landmarks were detected
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    
                    # Draw prediction on frame
                    H, W, _ = frame.shape
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, predicted_character, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Update prediction text
                    prediction_placeholder.success(f"Detected Sign: {predicted_character}")
                    
                except Exception as e:
                    prediction_placeholder.warning("Processing hand sign...")
            else:
                prediction_placeholder.info("No hand detected. Please show your hand to the camera.")
            
            # Display the frame
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Add a small delay to prevent high CPU usage
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Add some helpful information
    with st.expander("How to use"):
        st.markdown("""
            1. Allow camera access when prompted
            2. Make sure your hand is clearly visible in the frame
            3. Hold your hand sign steady
            4. The app will detect and display the recognized sign in real-time
        """)
    
    with st.expander("Supported Signs"):
        st.markdown("""
            The app recognizes:
            - All letters (A-Z)
            - Common phrases:
                - Hello
                - Thank You
                - I Love You
                - Sorry
                - Please
                - You're Welcome
                - Done
        """)

if __name__ == "__main__":
    main() 