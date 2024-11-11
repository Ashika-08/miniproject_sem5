from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
model = YOLO(r'C:\Users\Atimanyu\OneDrive\Documents\programs\Projects\miniprojectv2\runs\detect\train\weights\best.pt')
img = Image.open('test2.bmp')
results = model(img)
img_cv2 = cv2.imread('test2.bmp')
for result in results[0].boxes:
    # Get bounding box coordinates, confidence, and class
    x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
    conf = result.conf[0].cpu().numpy()
    cls = int(result.cls[0].cpu().numpy())

    # Draw the bounding box
    cv2.rectangle(img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color
    cv2.putText(img_cv2, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y1)-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# Display the result
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()  
# model.train(data=r'C:\Users\Atimanyu\OneDrive\Documents\programs\Projects\miniproject_sem5\dataset\data.yaml', epochs=200, imgsz=640,exist_ok = True)

