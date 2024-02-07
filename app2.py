import streamlit as st
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Define the main function for the Streamlit app
def main():
    # Set title of the Streamlit app
    st.title("Object Detection with YOLO")

    # Upload image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform object detection if an image is uploaded
    if uploaded_image is not None:
        try:
            # Convert uploaded image to numpy array
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

            # Perform object detection
            results = model(image)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Calculate person count
            person_count = sum(1 for class_id in detections.class_id if class_id == 0)  # Assuming class_id 0 corresponds to "person"

            # Initialize annotators
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # Annotate image
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            # Add text indicating the person count
            annotated_image = cv2.putText(annotated_image, f"Person Count: {person_count}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display annotated image
            st.image(annotated_image, channels="BGR", caption="Annotated Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
