import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Streamlit app
st.title("Object Detection with YOLO")

# Sidebar options
choice = st.selectbox("Select", ["Upload image"])
conf = st.number_input("Confidence threshold", 0.2)

if choice == "Upload image":
    # File uploader for image
    image_data = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if image_data is not None:
        img_summit_button = st.button("Predict")

        if img_summit_button:
            # Open the uploaded image
            image = Image.open(image_data)
            # Save the image temporarily
            image.save("input_data_image.png")

            # Read the image using OpenCV
            frame = cv2.imread("input_data_image.png")

            # Perform object detection
            results = model.predict(source=frame, iou=0.7, conf=conf)
            plot_show = results[0].plot()

            # Display the predicted image
            st.image(plot_show, caption="Predicted Image", use_column_width=True)

            # ------------------------------------------ Model predicted result all boxes ------------------------------------------#
            # get_array = results[0].boxes.numpy().boxes.tolist()

            # # Function to sort array
            # get_array.sort(key=sort_array_func)

            # for ind, i in enumerate(get_array):
            #     cv2.rectangle(frame_without_condition, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)
            #     cv2.putText(frame_without_condition, str(ind+1), (int(i[0])-70, int(i[1])+10), cv2.COLOR_BGR2GRAY, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
            #     cv2.line(frame_without_condition, pt1=(int(i[0]), int(i[1])+10), pt2=(int(i[0])-40, int(i[1])+5), color=(0, 0, 255), thickness=2)

            # count = str(len(get_array))
            # frame_without_condition = cv2.resize(frame_without_condition, (200, 750))
            # cv2.putText(frame_without_condition, "" + str(len(get_array)), (1, 680), cv2.COLOR_BGR2GRAY, 1, (255, 0, 0), 2)

            # classes_model = model.names
            # detected_list = [classes_model[i] for i in results[0].boxes.cls.tolist()]
