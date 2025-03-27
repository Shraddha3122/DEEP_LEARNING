import cv2
import numpy as np

# Start capturing video
cap = cv2.VideoCapture(0)

# Function to estimate basic emotions using mouth aspect ratio (MAR) with simple logic
def detect_emotion(roi_gray):
    height, width = roi_gray.shape

    # Define ROI (region of interest) for mouth region
    mouth_roi = roi_gray[int(height * 0.65): height, int(width * 0.25): int(width * 0.75)]

    # Calculate average intensity to detect smile (higher intensity indicates smile or open mouth)
    avg_intensity = np.mean(mouth_roi)

    if avg_intensity > 150:
        return "Happy 😀"
    elif avg_intensity < 100:
        return "Sad 😢"
    else:
        return "Neutral 🙂"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simulate a face detection rectangle in the center (default ROI)
    height, width = gray.shape
    x, y, w, h = int(width * 0.3), int(height * 0.3), int(width * 0.4), int(height * 0.4)

    # Draw rectangle to simulate detected face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract face region from gray image
    face_roi = gray[y:y + h, x:x + w]

    # Detect emotion using brightness (intensity)
    emotion = detect_emotion(face_roi)

    # Display detected emotion
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
