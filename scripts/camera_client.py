import cv2
import requests
import numpy as np
import time

# Configuration
API_URL = "http://localhost:8000/api/v1/recognize"
CAMERA_INDEX = 0

def run_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera started. Press 'q' to quit.")
    
    last_recognition_time = 0
    recognition_interval = 1.0 # Run recognition every 1 second to save CPU/Network
    current_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # Display existing bounding boxes
        for res in current_results:
            name = res["name"]
            conf = res["confidence"]
            box = res["box"]
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Facial Attendance Client", frame)

        # Trigger recognition periodically
        if current_time - last_recognition_time > recognition_interval:
            # Encode frame as JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            try:
                response = requests.post(API_URL, files=files, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    current_results = data.get("results", [])
                    last_recognition_time = current_time
            except Exception as e:
                print(f"Connection error: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
