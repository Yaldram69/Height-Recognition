import cv2
import numpy as np
# from ultralytics import YOLO
import ultralytics
from ultralytics import YOLO


# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Use a custom-trained model if available

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Reference object details (known real height in cm)
REFERENCE_HEIGHT_REAL = 50  # Example: known object height in cm
REFERENCE_HEIGHT_PIXELS = None  # To be calculated dynamically

# Define class ID for "person" (YOLOv8 uses COCO classes, where person = 0)
PERSON_CLASS_ID = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 segmentation
    results = model(frame)

    for result in results:
        masks = result.masks  # Segmentation masks
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs of detected objects

        if masks is not None:
            for i, mask in enumerate(masks.data):  # Iterate through detected masks
                if int(class_ids[i]) == PERSON_CLASS_ID:
                    continue  # Ignore the "person" class

                mask = mask.cpu().numpy()  # Convert to NumPy array
                mask = (mask * 255).astype(np.uint8)  # Convert to 0-255 range

                # Create a blank canvas for the mask
                mask_canvas = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
                mask_canvas[mask > 128] = 255  # Apply threshold

                # Find contours in the mask
                contours, _ = cv2.findContours(mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) > 5:  # Ensure it's a valid object
                        # Get bounding rectangle from contour
                        x, y, w, h = cv2.boundingRect(contour)

                        # Estimate real-world height
                        height_pixels = h  # Height in pixels
                        if REFERENCE_HEIGHT_PIXELS:
                            scale_factor = REFERENCE_HEIGHT_REAL / REFERENCE_HEIGHT_PIXELS
                            height_real = height_pixels * scale_factor
                        else:
                            height_real = "Calibrating..."

                        # Draw contour and bounding box
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Display height on frame
                        cv2.putText(frame, f"Height: {height_pixels}px / {height_real} cm",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Webcam - Segmentation-based Height Measurement", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
