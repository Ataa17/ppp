import cv2
import face_recognition
import os
import numpy as np

# Path to the folder of known faces
KNOWN_FACES_DIR = "known_faces"

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

known_encodings = []
known_names = []

# Load known face encodings
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].capitalize()
            known_names.append(name)
        else:
            print(f"[Warning] No face found in {filename}")

# Start video capture
cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for Haar detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append((y, x + w, y + h, x))

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Draw rectangle & label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
