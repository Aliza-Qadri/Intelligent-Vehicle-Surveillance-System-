import csv
import pandas as pd
from ultralytics import YOLO
import cv2
import easyocr
from google.colab.patches import cv2_imshow
from ocr_validation import validate_ocr  # Import the validation function

# Load pre-trained YOLO model
model = YOLO('/content/drive/MyDrive/license_plate_detector.pt')

# Path to your video file
video_path = '/content/drive/MyDrive/car 1-1.mp4'

# Read the video
video = cv2.VideoCapture(video_path)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])
detection_logs = []  # To store detection time and number plate details

while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit the loop when video ends

    # Crop the top-right corner for timestamp (adjust coordinates as needed)
    height, width, _ = frame.shape
    timestamp_region = frame[0:int(height * 0.1), int(width * 0.8):width]

    # Perform OCR on the timestamp region
    timestamp_result = reader.readtext(timestamp_region, detail=0)
    detected_time = timestamp_result[0] if timestamp_result else "Unknown Time"

    # Validate OCR for timestamp region
    validate_ocr(timestamp_region, detected_time)

    # Run YOLO model on the current frame
    results = model(frame)

    # Process YOLO results
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Crop the detected area (license plate)
        cr_img = frame[int(y1):int(y2), int(x1):int(x2)]

        # Perform OCR on the cropped image
        result = reader.readtext(cr_img)

        # Validate OCR for license plate
        ocr_res = []
        for k in range(len(result)):
            if (result[k][2]) > 0.3 and len(result[k][1]) > 2:
                ocr_res.append(result[k][1])
        ocr_res = ''.join(ocr_res).upper()
        validate_ocr(cr_img, ocr_res)

        if len(ocr_res) > 1:
            detection_logs.append({'Time': detected_time, 'Number Plate': ocr_res})
            # Draw the detection on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{ocr_res} ({detected_time})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with detections
    cv2_imshow(frame)

# Convert detection logs to a DataFrame
df = pd.DataFrame(detection_logs)
print(df)  # Display the DataFrame in the output

# Save detections to a CSV file
csv_file_path = '/content/detections.csv'
df.to_csv(csv_file_path, index=False)
print(f"\nDetection log saved to: {csv_file_path}")

# Release the video object
video.release()
cv2.destroyAllWindows()
