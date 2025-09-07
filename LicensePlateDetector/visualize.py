import ast
import cv2
import numpy as np
import pandas as pd

DEBUG = True  # Toggle this to False to silence debug output

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw fancy borders
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def parse_bbox(bbox_str):
    try:
        return ast.literal_eval(bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    except Exception as e:
        if DEBUG:
            print(f"Failed to parse bbox: {bbox_str} — {e}")
        return [0, 0, 0, 0]

# Load results
results = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# Preload license plate crops
license_plate = {}
for car_id in np.unique(results['car_id']):
    try:
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        best_row = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)].iloc[0]

        frame_index = best_row['frame_nmr']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            if DEBUG:
                print(f"Failed to read frame {frame_index} for car_id {car_id}")
            continue

        x1, y1, x2, y2 = parse_bbox(best_row['license_plate_bbox'])
        license_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': best_row['license_number']
        }
    except Exception as e:
        if DEBUG:
            print(f"Error processing car_id {car_id}: {e}")
        continue

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_nmr = -1

# Frame loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1

    df_ = results[results['frame_nmr'] == frame_nmr]
    if DEBUG:
        print(f"Frame {frame_nmr}: {len(df_)} detections")

    for _, row in df_.iterrows():
        car_id = row['car_id']
        car_x1, car_y1, car_x2, car_y2 = parse_bbox(row['car_bbox'])
        x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])

        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        try:
            license_crop = license_plate[car_id]['license_crop']
            plate_text = license_plate[car_id]['license_plate_number']
            H, W, _ = license_crop.shape

            # Check bounds before placing crop
            if int(car_y1) - H - 100 > 0:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                cv2.putText(frame, plate_text,
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
        except Exception as e:
            if DEBUG:
                print(f"Error overlaying license plate for car_id {car_id}: {e}")
            continue

    out.write(frame)
    resized_frame = cv2.resize(frame, (1280, 720))
    wait_time = int(1000 / fps) if fps > 0 else 30
    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
