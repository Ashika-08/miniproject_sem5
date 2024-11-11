import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Load your custom YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load the image
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
