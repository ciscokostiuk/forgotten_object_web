import cv2
import time
import json
import os
import smtplib
from email.mime.text import MIMEText
import requests
from datetime import datetime

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor", "laptop", "backpack", "handbag", "suitcase", "cell phone", "book"]

class ForgottenObjectDetector:
    def __init__(self, config_path='config.json'):
        self.load_config(config_path)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.object_timestamps = {}
        self.sent_alerts = set()
        self.log_file = "detections.log"
        os.makedirs("snapshots", exist_ok=True)
        self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
                                            'MobileNetSSD_deploy.caffemodel')

    def load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        self.min_area = config["min_area"]
        self.forgotten_time = config["forgotten_time"]
        self.video_source = config["video_source"]
        self.target_objects = config["target_objects"]
        self.email_notify = config.get("email_notify", "")
        self.telegram_token = config.get("telegram_token", "")
        self.telegram_chat_id = config.get("telegram_chat_id", "")

    def send_email(self, subject, body):
        if not self.email_notify:
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.email_notify
            msg["To"] = self.email_notify
            with smtplib.SMTP("localhost") as server:
                server.send_message(msg)
        except Exception as e:
            print("Email error:", e)

    def send_telegram(self, message, image_path=None):
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, data={"chat_id": self.telegram_chat_id, "text": message})
            if image_path:
                url_photo = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    requests.post(url_photo, data={"chat_id": self.telegram_chat_id}, files={"photo": photo})
        except Exception as e:
            print("Telegram error:", e)

    def log_detection(self, message):
        with open(self.log_file, "a") as log:
            log.write(f"{datetime.now()} - {message}\n")

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgmask = self.bg_subtractor.apply(frame)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            current_time = time.time()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx] if idx < len(CLASSES) else "unknown"

                    if label not in self.target_objects:
                        continue

                    box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                    (x1, y1, x2, y2) = box.astype("int")
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1

                    if w * h < self.min_area:
                        continue

                    center = (x + w // 2, y + h // 2)
                    key = f"{center[0]}-{center[1]}-{label}"

                    if key not in self.object_timestamps:
                        self.object_timestamps[key] = current_time

                    if current_time - self.object_timestamps[key] > self.forgotten_time:
                        if key not in self.sent_alerts:
                            msg = f"Забутий об'єкт: {label} ({x}, {y})"
                            self.log_detection(msg)
                            snapshot_path = f"snapshots/{label}_{int(current_time)}.jpg"
                            cv2.imwrite(snapshot_path, frame)
                            self.send_email("Забутий об'єкт", msg)
                            self.send_telegram(msg, snapshot_path)
                            self.sent_alerts.add(key)
                        cv2.putText(frame, f"Забутий: {label}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Виявлення забутих об'єктів", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()