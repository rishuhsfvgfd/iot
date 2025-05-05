import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
import time
import pandas as pd
import requests
import json
import os
import datetime

# Load YAMNet model from TensorFlow Hub
print("Loading YAMNet model...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
print("Model loaded successfully!")

# Get class names from the model
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path) as file:
    for line in file:
        class_names.append(line.strip().split(',')[2])
print(f"Total classes: {len(class_names)}")

# List of human-related sound classes
HUMAN_SOUNDS = [
    'Speech', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
    'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell', 'Children shouting',
    'Screaming', 'Whispering', 'Laughter', 'Chuckle, chortle', 'Giggle', 'Snicker',
    'Belly laugh', 'Chatter', 'Crying, sobbing', 'Baby cry, infant cry', 'Whimper',
    'Sigh', 'Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Male singing',
    'Female singing', 'Child singing', 'Synthetic singing', 'Rapping', 'Humming',
    'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 'Gasp', 'Pant',
    'Snort', 'Cough', 'Throat clearing', 'Sneeze', 'Sniff', 'Run', 'Shuffle', 'Walk, footsteps',
    'Finger snapping', 'Clapping', 'display_name'
]

# Add chainsaw and gun sounds
CHAINSAW_SOUNDS = ['Chainsaw', 'Power tool', 'Sawing', 'Vehicle', 'Crackle', 'Whale vocalization']
GUN_SOUNDS = ['Gunshot, gunfire', 'Machine gun', 'Cap gun', 'Artillery fire', 'Skateboard', 'Burst', 'pop', 'Bang', 'Firecracker', 'cap gun', 'Firearm', 'Weapons', 'Percussion', 'Slap', 'Whip', 'Cannon', 'Crack']

# Combined alert sounds
ALERT_SOUNDS = HUMAN_SOUNDS + CHAINSAW_SOUNDS + GUN_SOUNDS

# Audio parameters
SAMPLE_RATE = 16000  # YAMNet requires 16kHz mono audio
BLOCK_SIZE = 16000  # Process 1 second of audio at a time
CHANNELS = 1  # Mono audio

# ThingSpeak API settings
THINGSPEAK_API_KEY = "TBS9BPPWF1MO6HIO"  # Replace with your ThingSpeak write API key
THINGSPEAK_URL = "https://api.thingspeak.com/update"
THINGSPEAK_CHANNEL_ID = "2907484"  # Replace with your ThingSpeak channel ID

# Flask API settings for web client communication
FLASK_SERVER_PORT = 5001
ENABLE_FLASK_SERVER = True

# Queue for audio data
audio_queue = queue.Queue()

# Top predictions to display
TOP_K = 5

# Confidence threshold for alert detection
DETECTION_THRESHOLD = 0.5
DETECTION_THRESHOLD_CHAINSAW = 0.6
DETECTION_THRESHOLD_GUNSHOT = 0.2
# Flag to control sound detection
sound_detection_active = True

# Directory for saving detection logs
LOG_DIR = "detection_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create figure for visualization with more space and better layout
fig = plt.figure(figsize=(12, 16))
gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 0.5, 0.5], hspace=0.5)

# Top predictions bar chart
ax1 = fig.add_subplot(gs[0])
bars = ax1.barh(np.arange(TOP_K), np.zeros(TOP_K), align='center')
ax1.set_yticks(np.arange(TOP_K))
ax1.set_yticklabels([''] * TOP_K)
ax1.set_xlabel('Confidence')
ax1.set_title('YAMNet Predictions')
ax1.set_xlim(0, 1)

# Waveform display
ax2 = fig.add_subplot(gs[1])
times = np.arange(BLOCK_SIZE) / SAMPLE_RATE
line, = ax2.plot(times, np.zeros(BLOCK_SIZE))
ax2.set_ylim(-1, 1)
ax2.set_xlim(0, BLOCK_SIZE / SAMPLE_RATE)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Waveform')

# Alert status
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
alert_text = ax3.text(0.5, 0.5, 'No Alerts', ha='center', va='center', 
                     fontsize=20, color='green', transform=ax3.transAxes)

# Current sound display
ax4 = fig.add_subplot(gs[3])
ax4.axis('off')
current_sound_text = ax4.text(0.5, 0.5, 'Listening...', ha='center', va='center', 
                             fontsize=16, color='blue', transform=ax4.transAxes)

plt.tight_layout(pad=3.0)

# Store detection history
detection_history = []
recent_alerts = []  # For Flask API

# Log file for detections
log_filename = os.path.join(LOG_DIR, f"sound_detection_log_{time.strftime('%Y%m%d')}.csv")

# Function to initialize log file
def init_log_file():
    """Initialize or create the log file with headers if it doesn't exist"""
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            f.write("timestamp,sound_class,confidence,alert_type,thingspeak_status\n")
    return log_filename

# Function to log detection to file
def log_detection(sound_class, confidence, alert_type, thingspeak_status):
    """Log detection to CSV file"""
    with open(log_filename, 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp},{sound_class},{confidence},{alert_type},{thingspeak_status}\n")

# Function to upload detection to ThingSpeak
def upload_to_thingspeak(sound_class, confidence, alert_type):
    """Upload sound detection data to ThingSpeak"""
    
    # Map alert type to a numerical value for ThingSpeak
    # 1 = Normal sound, 2 = Human sound, 3 = Chainsaw sound, 4 = Gun sound
    alert_type_value = 1
    if alert_type == "HUMAN":
        alert_type_value = 2
    elif alert_type == "CHAINSAW":
        alert_type_value = 3
    elif alert_type == "GUN":
        alert_type_value = 4
    
    # Prepare data for ThingSpeak
    # Using field6 for sound detection type (1-4)
    # Using field8 for confidence (0.0-1.0) --Using it
    # Using field8 for sound class name (string) -- Not using now
    data = {
        'api_key': THINGSPEAK_API_KEY,
        'field6': alert_type_value,
        'field8': round(float(confidence), 2),
        # 'field8': sound_class
    }
    
    thingspeak_status = "Success"
    try:
        response = requests.post(THINGSPEAK_URL, data=data)
        if response.status_code == 200:
            print(f"Successfully uploaded data to ThingSpeak: {sound_class}, {alert_type}")
        else:
            print(f"Failed to upload to ThingSpeak. Status code: {response.status_code}")
            thingspeak_status = f"Failed ({response.status_code})"
    except Exception as e:
        print(f"Error uploading to ThingSpeak: {e}")
        thingspeak_status = f"Error: {str(e)[:50]}"
    
    # Log the detection regardless of ThingSpeak status
    log_detection(sound_class, confidence, alert_type, thingspeak_status)
    
    return thingspeak_status

# Flask server for web interface communication
if ENABLE_FLASK_SERVER:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    @app.route('/sound_status', methods=['GET'])
    def get_sound_status():
        """Return the current sound detection status and recent alerts"""
        return jsonify({
            'active': sound_detection_active,
            'recent_alerts': recent_alerts[-30:]  # Return last 30 alerts
        })
    
    @app.route('/sound_control', methods=['POST'])
    def control_sound_detection():
        """Enable or disable sound detection"""
        global sound_detection_active
        data = request.json
        if 'active' in data:
            sound_detection_active = bool(data['active'])
            print(f"Sound detection {'enabled' if sound_detection_active else 'disabled'} by web request")
            return jsonify({'success': True, 'active': sound_detection_active})
        return jsonify({'success': False, 'message': 'Invalid request'})
    
    @app.route('/detection_history', methods=['GET'])
    def get_detection_history():
        """Return the full detection history"""
        # Convert detection history to a more web-friendly format
        history = []
        for timestamp, sound, confidence, alert in detection_history:
            history.append({
                'timestamp': timestamp,
                'sound': sound,
                'confidence': float(confidence),
                'alert': bool(alert)
            })
        return jsonify(history)
        
    # Start Flask server in a separate thread
    def start_flask_server():
        app.run(host='0.0.0.0', port=FLASK_SERVER_PORT)
    
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    print(f"Flask server started on port {FLASK_SERVER_PORT}")

# Callback function for audio stream
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    # Skip processing if sound detection is disabled
    if not sound_detection_active:
        return
        
    # Convert to mono if needed and ensure correct format
    if CHANNELS > 1:
        indata = np.mean(indata, axis=1)
    audio_data = indata.flatten().astype(np.float32)
    
    # Normalize to [-1.0, 1.0] range as required by YAMNet
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    audio_queue.put(audio_data)

# Function to check if a sound should trigger an alert
def is_alert_sound(sound_class):
    """Check if the detected sound is in the alert sounds list."""
    return (sound_class in HUMAN_SOUNDS or 
            sound_class in CHAINSAW_SOUNDS or 
            sound_class in GUN_SOUNDS)

# Function to get the alert type
def get_alert_type(sound_class):
    """Determine the type of alert based on the sound class."""
    if sound_class in HUMAN_SOUNDS:
        return "HUMAN"
    elif sound_class in CHAINSAW_SOUNDS:
        return "CHAINSAW"
    elif sound_class in GUN_SOUNDS:
        return "GUN"
    return "NORMAL"

# Function to process audio and update visualization
def update_plot(frame):
    global recent_alerts
    
    try:
        # Get audio data from queue
        audio_data = audio_queue.get_nowait()
        
        # Update waveform display
        line.set_ydata(audio_data)
        
        # Process with YAMNet
        scores, embeddings, spectrogram = yamnet_model(audio_data)
        scores = scores.numpy()
        
        # Get top k predictions
        top_indices = np.argsort(scores[0])[-TOP_K:][::-1]
        top_scores = scores[0][top_indices]
        top_labels = [class_names[i] for i in top_indices]
        
        # Update bar chart
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            bar.set_width(score)
        ax1.set_yticklabels(top_labels)
        
        # Update current sound display
        current_sound = f"Now Hearing: {top_labels[0]} ({top_scores[0]:.2f})"
        current_sound_text.set_text(current_sound)
        
        # Check for alert sounds
        alert_detected = False
        alert_sounds_detected = []
        
        for sound, score in zip(top_labels, top_scores):
            if is_alert_sound(sound) and score > DETECTION_THRESHOLD:
                alert_detected = True
                alert_type = get_alert_type(sound)
                alert_sounds_detected.append((sound, score, alert_type))
            elif sound == 'Crackle' and score > DETECTION_THRESHOLD_CHAINSAW:
                alert_detected = True
                alert_type = get_alert_type(sound)  # Or use get_alert_type if appropriate
                alert_sounds_detected.append((sound, score, alert_type))
            elif sound == 'Crackle' and score > DETECTION_THRESHOLD_GUNSHOT and score < 0.4:
                alert_detected = True
                alert_type = "GUN"  # Or use get_alert_type if appropriate
                alert_sounds_detected.append((sound, score, alert_type))
        
        # Update alert detection status
        if alert_detected:
            highest_alert_sound = alert_sounds_detected[0]
            alert_text.set_text(f"⚠️ ALERT: {highest_alert_sound[2]}")
            alert_text.set_color('red')
            alert_message = f"⚠️ {highest_alert_sound[2]} ALERT [{time.strftime('%H:%M:%S')}]: {highest_alert_sound[0]} (Confidence: {highest_alert_sound[1]:.2f})"
            print(alert_message)
            
            # Add to recent alerts for web interface
            recent_alerts.append({
                'type': highest_alert_sound[2],
                'sound': highest_alert_sound[0],
                'confidence': float(highest_alert_sound[1]),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Keep only the most recent 50 alerts
            if len(recent_alerts) > 50:
                recent_alerts = recent_alerts[-50:]
                
            # Upload to ThingSpeak
            thingspeak_status = upload_to_thingspeak(
                highest_alert_sound[0],       # sound class
                highest_alert_sound[1],       # confidence
                highest_alert_sound[2]        # alert type
            )
        else:
            # Normal sound - still log it to ThingSpeak periodically (every 30 seconds)
            current_time = time.time()
            if not hasattr(update_plot, 'last_normal_upload') or current_time - update_plot.last_normal_upload > 30:
                update_plot.last_normal_upload = current_time
                # Upload normal sound
                thingspeak_status = upload_to_thingspeak(
                    top_labels[0],     # top detected sound
                    top_scores[0],     # confidence
                    "NORMAL"           # alert type
                )
            
            alert_text.set_text('No Alerts')
            alert_text.set_color('green')
        
        # Store detection in history
        current_time = time.strftime("%H:%M:%S")
        detection_history.append((current_time, top_labels[0], top_scores[0], alert_detected))
        
        # Limit history size to prevent memory issues
        if len(detection_history) > 1000:
            detection_history.pop(0)
        
        # Convert bars to a list before adding line, alert text, and current sound text
        return list(bars) + [line] + [alert_text] + [current_sound_text]
    except queue.Empty:
        # No audio data available
        if sound_detection_active:
            current_sound_text.set_text("Listening...")
        else:
            current_sound_text.set_text("Sound Detection Paused")
        
        # Convert bars to a list before adding line, alert text, and current sound text
        return list(bars) + [line] + [alert_text] + [current_sound_text]

# Initialize the timestamp for normal sound uploads
update_plot.last_normal_upload = time.time()

# Initialize log file
init_log_file()

# Start audio stream
def start_stream():
    with sd.InputStream(device=DEVICE_INDEX, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
                        channels=CHANNELS, callback=audio_callback):
        print(f"Audio stream started using device {DEVICE_INDEX}.")
        print("System is now monitoring for sounds...")
        print("ALERTS will be triggered for: HUMAN, CHAINSAW, and GUN sounds")
        print(f"Sound detection is currently {'ACTIVE' if sound_detection_active else 'INACTIVE'}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Audio stream stopped.")
            show_detection_summary()

# Function to show detection summary
def show_detection_summary():
    """Show a summary of all detections."""
    if detection_history:
        df = pd.DataFrame(detection_history, columns=['Time', 'Sound', 'Confidence', 'Alert Triggered'])
        print("\n=== Detection Summary ===")
        print(df)
        
        # Count occurrences of each sound
        sound_counts = df['Sound'].value_counts()
        print("\nMost Common Sounds:")
        print(sound_counts.head(10))
        
        # Count alert sound detections
        alert_detections = df[df['Alert Triggered'] == True]
        print(f"\nAlert sounds detected: {len(alert_detections)} out of {len(df)} total detections")
        if not alert_detections.empty:
            print("\nAlert Sound Detections:")
            print(alert_detections)
        
        # Export to CSV
        export_filename = os.path.join(LOG_DIR, f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(export_filename, index=False)
        print(f"\nDetection summary exported to {export_filename}")

# List available audio devices
print("Available audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"Device {i}: {device['name']}")

# Ask user to select an input device
device_id = int(input("Enter the device ID for your phone's audio input: "))
DEVICE_INDEX = device_id  # Use the selected device

# Check if ThingSpeak API key is set
if not THINGSPEAK_API_KEY or THINGSPEAK_API_KEY == "H3CF3ZUMBH9F2JGD":
    print("WARNING: Using default ThingSpeak API key. Replace with your actual API key for production use.")

# Start audio processing in a separate thread
audio_thread = threading.Thread(target=start_stream)
audio_thread.daemon = True
audio_thread.start()

# Start animation
ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
plt.show()