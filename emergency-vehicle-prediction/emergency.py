import cv2
import requests
import json
import numpy as np

ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/emergency-03jrm/1?api_key=ExnaXttg8l8qi2Ka6g1N"
IMAGE_SIZE = 640 
video_capture = cv2.VideoCapture(0) 
def detect_objects(frame):
    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    encoded_image = cv2.imencode('.jpg', resized_frame)[1].tobytes()
    response = requests.post(ROBOFLOW_MODEL_URL, files={"file": encoded_image})
    predictions = response.json().get("predictions", [])
    for prediction in predictions:
        x = int(prediction["x"] * width / IMAGE_SIZE)
        y = int(prediction["y"] * height / IMAGE_SIZE)
        w = int(prediction["width"] * width / IMAGE_SIZE)
        h = int(prediction["height"] * height / IMAGE_SIZE)
        label = prediction["class"]
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame

def main():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        detected_frame = detect_objects(frame)
        
        cv2.imshow('Live Object Detection', detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()