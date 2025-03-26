# Import required libraries
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
try:
    emotion_model = load_model('models/emotion_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 📸 Analyze Facial Expressions
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided. Please upload an image.'})

        # Read and decode the image
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected for emotion analysis.'})

        emotions = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48)) / 255.0
            face_array = np.expand_dims(face_resized, axis=(0, -1))
            
            # Predict emotion
            prediction = emotion_model.predict(face_array)
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = round(np.max(prediction) * 100, 2)

            emotions.append({'emotion': emotion, 'confidence': confidence})

        return jsonify({'emotions': emotions})

    except Exception as e:
        return jsonify({'error': str(e)})

# 👀 Analyze Eye Movements
@app.route('/analyze_eye', methods=['POST'])
def analyze_eye():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided. Please upload an image.'})

        # Read and decode the image
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected for eye analysis.'})

        eye_status = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detect eyes within the face ROI
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            detected_eyes = eyes.detectMultiScale(roi_gray)

            if len(detected_eyes) == 0:
                eye_status.append({'eye_status': 'Closed'})
            else:
                eye_status.append({'eye_status': 'Open'})

        return jsonify({'eye_status': eye_status})

    except Exception as e:
        return jsonify({'error': str(e)})

# 🧠 Analyze Micro-Expressions (Placeholder)
@app.route('/analyze_micro', methods=['POST'])
def analyze_micro():
    # Placeholder logic for micro-expression detection
    return jsonify({'status': 'Micro-expression analysis coming soon'})

# 🔥 Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running successfully'}), 200

# 🚀 Run Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
