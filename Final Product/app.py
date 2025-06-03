from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import os
import numpy as np

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
known_encodings = []
known_names = []

def load_known_faces():
    known_encodings.clear()
    known_names.clear()
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
load_known_faces()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = None
            access_text = "Access Denied"
            color = (0, 0, 255)  # Red

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    access_text = "Access Granted"
                    color = (0, 255, 0)  # Green

            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw "Access Granted/Denied" at top of the box
            cv2.putText(frame, access_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw name at bottom center if access is granted
            if access_text == "Access Granted" and name:
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = left + (right - left - text_size[0]) // 2
                text_y = bottom + text_size[1] + 10
                cv2.putText(frame, name.capitalize(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        file = request.files['image']
        name = request.form['name']
        if file and name:
            path = os.path.join(KNOWN_FACES_DIR, f"{name.lower()}.jpg")
            file.save(path)
            load_known_faces()
            return redirect(url_for('index'))
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
