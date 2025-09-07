import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize

# --- CONFIG ---
image_path = 'car6.jpg'
yolo_model_path = 'yolov8n.pt'  # Replace with your fine-tuned plate detector if available
svc_model_path = 'finalized_model.sav'
char_size = (20, 20)

# --- LOAD MODELS ---
yolo_model = YOLO(yolo_model_path)
svc_model = joblib.load(svc_model_path)

# --- STEP 1: Detect License Plate with YOLO ---
results = yolo_model(image_path)
boxes_data = results[0].boxes.data.cpu().numpy()

# Filter for license plate class (if applicable)
# Replace `0` with your license plate class index
plate_boxes = [box for box in boxes_data if int(box[5]) == 0]

if not plate_boxes:
    print("No license plate detected.")
    exit()

# Use first detected plate
x1, y1, x2, y2 = map(int, plate_boxes[0][:4])

# --- STEP 2: Crop Plate Region ---
img = cv2.imread(image_path)
plate_img = img[y1:y2, x1:x2]
gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

# --- STEP 3: Segment Characters using CCA ---
thresh_val = threshold_otsu(gray_plate)
binary_plate = (gray_plate > thresh_val).astype(np.uint8) * 255
label_img = measure.label(binary_plate)

# Filter and sort regions left-to-right
regions = [r for r in regionprops(label_img) if r.area > 50]
regions = sorted(regions, key=lambda r: r.bbox[1])  # Sort by x (minc)

characters = []
for region in regions:
    minr, minc, maxr, maxc = region.bbox
    h, w = maxr - minr, maxc - minc
    aspect_ratio = w / h
    if 0.2 < aspect_ratio < 1.5:
        char_img = gray_plate[minr:maxr, minc:maxc]
        resized_char = resize(char_img, char_size, anti_aliasing=True)
        characters.append(resized_char.flatten())

# --- STEP 4: Predict Characters with SVC ---
if not characters:
    print("No characters segmented.")
    exit()

characters_np = np.array(characters)
plate_text = ''.join(svc_model.predict(characters_np))

# --- STEP 5: Display Results ---
print(f" License Plate Detected: {plate_text}")

# Optional: Show segmented characters
for i, char in enumerate(characters):
    char_img = np.reshape(char, char_size)
    cv2.imshow(f'Char {i}', char_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
