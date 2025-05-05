#include <SPI.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <TimeLib.h>
#include <WiFiUdp.h>
#include <NTPClient.h>

// RFID Pin Definitions
#define RST_PIN 25    // RST pin on GPIO25
#define SS_PIN 5      // SS/SDA pin on GPIO5

// Sensor Pins
const int pirPin = 13;           // PIR sensor
const int trigPin = 12;          // Ultrasonic sensor trigger
const int echoPin = 14;          // Ultrasonic sensor echo
const int LED_PIN = 32;          // LED on GPIO32
const int RFID_LED_PIN = 27;     // RFID LED on GPIO27
const int BUZZER_PIN = 33;       // Buzzer on GPIO33
const int PIR_TEST_PIN=15;

// WiFi Credentials
const char* ssid = "GuptaJi";
const char* password = "Apple@123";

// ThingSpeak HTTP Settings (for RFID)
unsigned long channelID = 2907484;
const char* writeAPIKey = "TBS9BPPWF1MO6HIO";

// ThingSpeak MQTT Configuration (for Detection)
const char* mqtt_server = "mqtt3.thingspeak.com";
const int mqtt_port = 1883;
const char* mqtt_client_id = "HS0tGBUyDwICMykQFjIHHQU";
const char* mqtt_username = "HS0tGBUyDwICMykQFjIHHQU";
const char* mqtt_password = "KmRvUv6VQHKDNFPg700H370i";
const char* channel_topic = "channels/2907484/publish";

// Camera Configuration
const char* camera_ip = "192.168.105.161";
const int camera_port = 8080;

// Detection Thresholds
const float maxDistance = 50.0;  // Maximum distance in cm
const int detectionCooldown = 3000;  // 3 seconds cooldown
const int alarmBlinks = 3;       // Number of alarm blinks/beeps
const int alarmOnTime = 300;     // Alarm on duration (ms)
const int alarmOffTime = 300;    // Alarm off duration (ms)

// Variables
String lastUID = "";             // Track last RFID UID
bool motionDetected = false;
bool objectDetected = false;
unsigned long lastDetectionTime = 0;
float distance = 0;

// Object Instances
MFRC522 rfid(SS_PIN, RST_PIN);
WiFiClient espClient;
PubSubClient client(espClient);
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org");

// Function Prototypes
void setupWiFi();
void reconnectMQTT();
float measureDistance();
void triggerAlarm();
void activateCamera();
void readRFID();

void IRAM_ATTR handlepir(){
  pirsense=true;
}

void setup() {
  Serial.begin(115200);

  // Initialize RFID
  SPI.begin(18, 19, 23, 5); // SCK (GPIO18), MISO (GPIO19), MOSI (GPIO23), SS (GPIO5)
  rfid.PCD_Init();
  Serial.println("RFID Reader Initialized");

  // Initialize Sensor Pins
  pinMode(pirPin, INPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(RFID_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_TEST_PIN,OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(RFID_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(PIR_TEST_PIN,LOW);

  // Connect to WiFi
  setupWiFi();

  // Configure MQTT
  client.setServer(mqtt_server, mqtt_port);

  // Initialize Time Client
  timeClient.begin();
  timeClient.setTimeOffset(0); // Set timezone offset in seconds

  attachInterrupt(digitalPinToInterrupt(PIR_PIN), handlePIR, RISING);

  esp_sleep_enable_ext0_wakeup((gpio_num_t)PIR_PIN, 1);

  Serial.println("System Initialized");
}

void loop() {
  // Maintain MQTT Connection
  if (!client.connected()) {
    reconnectMQTT();
  }
  client.loop();

  // Update Time
  timeClient.update();

  // Read RFID Tags
  readRFID();

  // Check Ultrasonic Sensor
  distance = measureDistance();
  objectDetected = distance > 0 && distance < maxDistance;

  // // Check PIR Sensor
  // motionDetected = digitalRead(pirPin) == HIGH;

  // Handle Detection
  if (motionDetected) {
    digitalWrite(PIR_TEST_PIN,HIGH);
    delay(1000);
    digitalWrite(PIR_TEST_PIN,LOW);
    Serial.println("Living object detected!");
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    // Send Initial Detection Alert
    DynamicJsonDocument doc(256);
    doc["detection"] = true;
    doc["distance"] = distance;
    doc["time"] = timeClient.getEpochTime();
    String payload = String("field1=1&field2=") + String(distance) +
                     "&field3=" + String(timeClient.getEpochTime()) +
                     "&status=DETECTION_ALERT";
    if (client.publish(channel_topic, payload.c_str())) {
      Serial.println("Initial detection alert sent to ThingSpeak");
    } else {
      Serial.println("Failed to send detection alert");
    }

    // Activate Camera
    activateCamera();

    // Update Last Detection Time
    lastDetectionTime = millis();
  }

  Serial.println("I am going to sleep");

  esp_light_sleep_start();

  Serial.println("Woke up from sleep");

  delay(1000); // Prevent flooding
}

void setupWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnectMQTT() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect(mqtt_client_id, mqtt_username, mqtt_password)) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void readRFID() {
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) {
    return;
  }

  // Get UID
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();

  // Send New UIDs to ThingSpeak via MQTT
  if (uid != lastUID) {
    Serial.print("UID: ");
    Serial.println(uid);
    
    if (client.connected()) {
      String payload = "field7=" + uid + "&status=RFID_DETECTED";
      
      if (client.publish(channel_topic, payload.c_str())) {
        Serial.println("RFID data sent to ThingSpeak via MQTT");
        lastUID = uid;
        
        // LED feedback for successful RFID read
        digitalWrite(RFID_LED_PIN, HIGH);
        delay(1000);  // Short blink
        digitalWrite(RFID_LED_PIN, LOW);
      } else {
        Serial.println("Failed to send RFID data via MQTT");
      }
    } else {
      Serial.println("MQTT not connected. Attempting reconnection...");
      reconnectMQTT();
    }
  }

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
  delay(800);  
}

float measureDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH);
  float distance = (duration * 0.0343) / 2;
  return distance;
}

void triggerAlarm() {
  Serial.println("Triggering alarm - human detected!");
  for (int i = 0; i < alarmBlinks; i++) {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(alarmOnTime);
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    delay(alarmOffTime);
  }
}

void activateCamera() {
  Serial.println("Activating camera for ML processing...");
  WiFiClient pythonClient;
  if (!pythonClient.connect("192.168.105.203", 8000)) {
    Serial.println("Connection to Python server failed");
    return;
  }

  pythonClient.println("DETECT");
  unsigned long timeout = millis();
  while (pythonClient.available() == 0) {
    if (millis() - timeout > 10000) {
      Serial.println("Python server timeout");
      pythonClient.stop();
      return;
    }
  }

  String response = pythonClient.readStringUntil('\n');
  Serial.print("Python response: ");
  Serial.println(response);

  DynamicJsonDocument doc(512);
  DeserializationError error = deserializeJson(doc, response);
  if (error) {
    Serial.print("JSON parsing failed: ");
    Serial.println(error.c_str());
    return;
  }

  bool isHuman = doc["isHuman"].as<bool>();
  String imageUrl = doc["imageUrl"].as<String>();
  String timestamp = doc["timestamp"].as<String>();
  Serial.print("Detection result - Human: ");
  Serial.println(isHuman ? "Yes" : "No");

  if (isHuman) {
    triggerAlarm();
  }

  String payload = String("field1=") + String(isHuman ? 2 : 1) +
                   "&field2=" + String(distance) +
                   "&field3=" + String(timeClient.getEpochTime()) +
                   "&field4=" + String(isHuman ? "human" : "animal") +
                   "&field5=" + imageUrl +
                //    "&field8=" + String(isHuman ? 1 : 0) +
                   "&status=" + String(isHuman ? "HUMAN_DETECTED" : "ANIMAL_DETECTED");
  if (client.publish(channel_topic, payload.c_str())) {
    Serial.println("Detection details sent to ThingSpeak");
  } else {
    Serial.println("Failed to send detection details");
  }

  pythonClient.stop();
}