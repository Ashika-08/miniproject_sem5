import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load the custom YOLOv8 model
model = YOLO(r'C:\Users\Atimanyu\OneDrive\Documents\programs\Projects\miniprojectv2\runs\detect\train\weights\best.pt')

# Upload image
uploaded_image = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image:
    # Load the uploaded image
    image = Image.open(uploaded_image)

    # Run prediction on the uploaded image
    results = model(image)

    # Check if there are any detections
    if not results[0].boxes:
        # If no detections, display a message
        st.warning("No detections found in the uploaded image.")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        # Convert image to OpenCV format (for annotation)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Annotate the image with bounding boxes
        for result in results[0].boxes:
            # Get bounding box coordinates, confidence, and class
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
            conf = result.conf[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())

            # Draw the bounding box
            cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color
            cv2.putText(img_cv2, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert annotated image from BGR to RGB for display in Streamlit
        annotated_image = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # Display original and annotated images in Streamlit
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)

        # Extract unique detected issues for recommendations
        detected_issues = set([model.names[int(result.cls[0].cpu().numpy())] for result in results[0].boxes])

        # Define recommendations for each detected issue
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

        # Display recommendations based on detected issues
        for issue in detected_issues:
            st.subheader(f"Recommendations for {issue.capitalize()}")
            for product in recommendations.get(issue, []):
                st.write(f"{product['product_name']}: {product['description']}")
