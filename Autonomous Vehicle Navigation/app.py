import cv2
import numpy as np

# 🎯 Load HOG for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 🎥 Load Uploaded Video
video_path = "D:/WebiSoftTech/D. DEEP LEARNING/OPEN_CV/Autonomous Vehicle Navigation/5921059-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# ⚡ Real-time Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # 🚶 Detect Pedestrians using HOG + SVM
    pedestrians, _ = hog.detectMultiScale(frame_resized, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Draw bounding boxes for pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_resized, "Pedestrian", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 🚗 Detect Vehicles using Contours
    blurred_frame = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1500 < area < 30000:  # Filter contour size to detect medium to large objects
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 1.0 < aspect_ratio < 3.0:  # Approximate aspect ratio for vehicles
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_resized, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🚦 Detect Traffic Signs (Contour-based with Circular Shape Detection)
    blurred_frame = cv2.GaussianBlur(frame_resized, (11, 11), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for traffic signs (red, blue, yellow)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    mask_combined = mask_red | mask_blue | mask_yellow

    # Find contours for traffic signs
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Check if aspect ratio is approximately square or circular
            if 0.8 <= aspect_ratio <= 1.2:
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame_resized, "Traffic Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 🖥️ Show output frame
    cv2.imshow("Real-Time Object Detection", frame_resized)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
