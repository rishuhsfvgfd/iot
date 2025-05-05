import cv2
import numpy as np
from ultralytics import YOLO
import requests
import datetime
import socket
import json
import os
import time
import threading
from flask import Flask, jsonify, request, send_file, render_template, send_from_directory
import paho.mqtt.client as mqtt

# Flask app for communication with ESP32 and serving images
app = Flask(__name__)

IP_WEBCAM_URL = "http://192.168.105.161:8080/video"  # Your IP Webcam URL
DETECTION_FOLDER = "detections"
MODEL_PATH = "model/human_animal_detector.h5"  # Path to your ML model
THINGSPEAK_HOST = "mqtt3.thingspeak.com"
THINGSPEAK_PORT = 1883
THINGSPEAK_CLIENT_ID = "Lz0xHDYeAB4sERYABAYzHgk"
THINGSPEAK_USERNAME = "Lz0xHDYeAB4sERYABAYzHgk"
THINGSPEAK_API_KEY = "6n1Cq8Ilh2MZOZorEkrFcQDC"
THINGSPEAK_CHANNEL_ID = "2907484"
THINGSPEAK_WRITE_API_KEY = "TBS9BPPWF1MO6HIO"

# New ports for socket server and Flask app
SOCKET_SERVER_PORT = 8000  # Changed from 5000
FLASK_APP_PORT = 8080      # Changed from 5001

# Ensure detection folder exists
if not os.path.exists(DETECTION_FOLDER):
    os.makedirs(DETECTION_FOLDER)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

# MQTT client setup
mqtt_client = mqtt.Client(client_id=THINGSPEAK_CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
mqtt_client.username_pw_set(THINGSPEAK_USERNAME, THINGSPEAK_API_KEY)

# Connect to MQTT broker
def connect_mqtt():
    try:
        mqtt_client.connect(THINGSPEAK_HOST, THINGSPEAK_PORT, 60)
        mqtt_client.loop_start()
        print("Connected to ThingSpeak MQTT broker")
        return True
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return False

# Function to access IP Webcam
def get_ip_webcam_frame():
    try:
        cap = cv2.VideoCapture(IP_WEBCAM_URL)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
            return frame
        else:
            print("Failed to get frame from IP Webcam")
            return None
    except Exception as e:
        print(f"Error accessing IP Webcam: {e}")
        return None
    
@app.route('/')
def index():
    return render_template('index.html')

# Detect human or animal using YOLOv8
def detect_human_or_animal(image):
    results = model(image, conf=0.5)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:  # Human
                return True
            elif cls in [15, 16, 17, 18, 19, 20, 21, 22, 23] and conf > 0.5:  # Animals (COCO dataset)
                return False
    return False  # Default to animal if no human detected

# Save detection image and metadata
def save_detection(image, is_human):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    detection_type = "human" if is_human else "not_human"
    filename = f"{detection_type}_{timestamp}.jpg"
    filepath = os.path.join(DETECTION_FOLDER, filename)
    cv2.imwrite(filepath, image)
    metadata = {
        "timestamp": timestamp,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": detection_type,
        "filename": filename,
        "filepath": filepath,
    }
    return metadata, filepath

# Send detection to ThingSpeak
def send_to_thingspeak(metadata, image_path):
    try:
        detection_type = 2 if metadata["type"] == "human" else 1
        topic = f"channels/{THINGSPEAK_CHANNEL_ID}/publish"
        payload = (
            f"field1={detection_type}&"
            f"field4={metadata['type']}&"
            f"field3={int(time.time())}&"
            f"status={metadata['type'].upper()}_DETECTED&"
            # f"field6={metadata['datetime']}"
        )
        result = mqtt_client.publish(topic, payload)
        if result.rc == 0:
            print(f"Detection sent to ThingSpeak: {metadata['type']}")
            upload_image_to_thingspeak(image_path, metadata)
            return True
        else:
            print(f"Failed to publish to ThingSpeak: {result.rc}")
            return False
    except Exception as e:
        print(f"Error sending to ThingSpeak: {e}")
        return False

# Upload image to ThingSpeak
def upload_image_to_thingspeak(image_path, metadata):
    try:
        # Construct URL with just the filename
        filename = os.path.basename(image_path)
        image_url = f"/detections/{filename}"  # Simplified URL
        topic = f"channels/{THINGSPEAK_CHANNEL_ID}/publish"
        payload = f"field5={image_url}&status=IMAGE_URL_UPDATED"
        mqtt_client.publish(topic, payload)
        return True
    except Exception as e:
        print(f"Error uploading image: {e}")
        return False

# Route to serve detection images
@app.route('/detections/<filename>')
def serve_detection(filename):
    return send_file(os.path.join(DETECTION_FOLDER, filename))

@app.route('/captured_images/<path:filename>')
def serve_captured_image(filename):
    """Serve captured images"""
    return send_from_directory('detections', filename)

# Route to trigger detection
@app.route('/detect', methods=['GET', 'POST'])
def trigger_detection():
    frame = get_ip_webcam_frame()
    if frame is None:
        return jsonify({"error": "Failed to get camera frame"}), 400
    is_human = detect_human_or_animal(frame)
    metadata, filepath = save_detection(frame, is_human)
    send_to_thingspeak(metadata, filepath)
    return jsonify({
        "isHuman": is_human,
        "timestamp": metadata["datetime"],
        "imageUrl": f"/detections/{os.path.basename(filepath)}",
        "type": metadata["type"]
    })

# Route for ESP32 to trigger detection
@app.route('/esp32_trigger', methods=['POST'])
def esp32_trigger():
    return trigger_detection()

# Get all detections
@app.route('/all_detections')
def get_all_detections():
    detections = []
    for filename in os.listdir(DETECTION_FOLDER):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            detection_type = parts[0]
            timestamp = '_'.join(parts[1:]).replace('.jpg', '')
            detections.append({
                "type": detection_type,
                "timestamp": timestamp,
                "datetime": datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                "image": f"/detections/{filename}"
            })
    return jsonify(detections)

# Socket server for ESP32 communication
def start_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', SOCKET_SERVER_PORT))  # Using new port
    server_socket.listen(5)
    print(f"Socket server started on port {SOCKET_SERVER_PORT}")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        data = client_socket.recv(1024).decode('utf-8').strip()
        if data == "DETECT":
            print("Detection requested via socket")
            frame = get_ip_webcam_frame()
            if frame is not None:
                is_human = detect_human_or_animal(frame)
                metadata, filepath = save_detection(frame, is_human)
                send_to_thingspeak(metadata, filepath)
                response = {
                    "isHuman": is_human,
                    "timestamp": metadata["datetime"],
                    "imageUrl": f"/detections/{os.path.basename(filepath)}",
                    "type": metadata["type"]
                }
                client_socket.send((json.dumps(response) + "\n").encode('utf-8'))
            else:
                client_socket.send('{"error": "Failed to get camera frame"}\n'.encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    connect_mqtt()
    socket_thread = threading.Thread(target=start_socket_server)
    socket_thread.daemon = True
    socket_thread.start()
    app.run(host='0.0.0.0', port=FLASK_APP_PORT, debug=True, use_reloader=False)