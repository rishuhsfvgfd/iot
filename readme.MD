# IoT-Based Wildlife and RFID Tracking System

This project is an IoT-based system designed to monitor wildlife activity and track RFID tags. It integrates hardware components like ESP32, ultrasonic sensors, and RFID readers with software for sound and visual detection, providing real-time monitoring and data visualization.

## Project Summary

The system performs the following tasks:
- **Sound Detection**: Identifies specific sounds (e.g., human voices, chainsaws, gunshots) using machine learning.
- **Visual Detection**: Tracks moving living objects using PIR and calculates distances using ultrasonic sensors and a camera module.
- **RFID Tracking**: Monitors RFID tags to detect their presence or absence.
- **Data Visualization**: Displays detection history and trends on a web-based dashboard.
- **Low Power Mode**: Implements sleep mode on the ESP32 to conserve power.



## How to Run the Code

### 1. **Hardware Setup**
- Connect the ESP32 to the ultrasonic sensor, PIR sensor, RFID reader, and camera module.
- Ensure the hardware is powered and connected to your WiFi network.

### 2. **Run the ESP32 Code**
- Open the `esp_withsleep.ino` file in the Arduino IDE.
- Configure your WiFi credentials and ThingSpeak API keys in the code.
- Upload the code to the ESP32.

### 3. **Run the Machine Learning Models**
- Navigate to the `camera_model` 
  cd iot/camera_model
  python camera_ml.py
- Navigate to the `sound_model` 
  cd iot/sound_model
  python sound_ml.py

### 4. **Dashboard**
- Run dashboard.html in a browser or visit https://rishuhsfvgfd.github.io/iot/dashboard.html