import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt

# Load your custom YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Run prediction
    results = model.predict(image)

    # Draw bounding boxes on the image
    annotated_image = results[0].plot()  # results[0] gives the first result with bounding boxes

    # Display original and annotated images in Streamlit
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)

    # Extract unique detected issues
    detected_issues = set([result.label for result in results[0].boxes.data])

    # Recommendations for each detected issue
    recommendations = {
        "acne": [
            {"product_name": "Salicylic Acid Cleanser", "description": "Reduces acne and clears pores."},
            {"product_name": "Benzoyl Peroxide Gel", "description": "Helps reduce acne-causing bacteria."}
        ],
        "eye_bags": [
            {"product_name": "Caffeine Eye Cream", "description": "Reduces puffiness and dark circles."}
        ],
        "skin_redness": [
            {"product_name": "Aloe Vera Gel", "description": "Soothes and calms irritated skin."}
        ]
    }

    # Display recommendations
    for issue in detected_issues:
        st.subheader(f"Recommendations for {issue.capitalize()}")
        for product in recommendations.get(issue, []):
            st.write(f"{product['product_name']}: {product['description']}")

    # Convert the image to OpenCV format for additional processing (optional)
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes on the OpenCV image for visualization
    for result in results[0].boxes:
        # Get bounding box coordinates, confidence, and class
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
        conf = result.conf[0].cpu().numpy()
        cls = int(result.cls[0].cpu().numpy())

        # Draw the bounding box
        cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color
        cv2.putText(img_cv2, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert BGR to RGB for display in Streamlit
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Display the result
    st.image(img_rgb, caption="Processed Image with Bounding Boxes", use_column_width=True)
else:
    st.warning("Please upload an image to proceed.")
