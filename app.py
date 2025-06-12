from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import io
import os

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model('pose_classifier_model.keras')

labels = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
POSE_NET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def extract_keypoints(landmarks):
    return [coord for idx in POSE_NET_INDICES for lm in [landmarks.landmark[idx]] for coord in (lm.x, lm.y)]

@app.route("/video_feed", methods=["POST"])
def video_feed():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    in_memory_file = file.read()
    frame = cv2.imdecode(np.frombuffer(in_memory_file, np.uint8), cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return jsonify({"error": "Failed to encode image"}), 500

    return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/jpeg")

@app.route("/predict_pose", methods=["POST"])
def predict_pose():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    in_memory_file = file.read()
    frame = cv2.imdecode(np.frombuffer(in_memory_file, np.uint8), cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if not results.pose_landmarks:
        return jsonify({"prediction": None, "confidence": 0})

    keypoints = extract_keypoints(results.pose_landmarks)
    input_data = np.array(keypoints).reshape(1, -1)
    preds = model.predict(input_data)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    predicted_label = labels[class_idx]

    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
