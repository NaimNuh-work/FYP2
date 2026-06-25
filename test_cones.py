import cv2
import numpy as np
import os
from pyproj import Transformer
from ultralytics import YOLO

# ==========================================
# RTSP OPTIMIZATION
# ==========================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# ==========================================
# MATHEMATICAL MAPPING ENGINE (FLAT GROUND)
# ==========================================
def calculate_direction_vector(u, v, img_width, img_height, hov_deg, tilt_deg, azimuth_deg):
    cx = img_width / 2.0
    cy = img_height / 2.0
    focal_length = cx / np.tan(np.radians(hov_deg) / 2.0)

    img_x = u - cx
    img_y = cy - v

    vec = np.array([img_x, focal_length, img_y])
    vec = vec / np.linalg.norm(vec)

    tilt = np.radians(tilt_deg)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(-tilt), -np.sin(-tilt)],
        [0, np.sin(-tilt), np.cos(-tilt)]
    ])

    azimuth = np.radians(-azimuth_deg)
    R_z = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])

    return R_z @ R_x @ vec


def calculate_flat_ground_intersection(cam_x, cam_y, cam_z, direction_vector, ground_elevation):
    v_z = direction_vector[2]

    if v_z >= 0:
        return None, None

    t = (ground_elevation - cam_z) / v_z

    hit_x = cam_x + (t * direction_vector[0])
    hit_y = cam_y + (t * direction_vector[1])

    return hit_x, hit_y


# ==========================================
# COLOR VALIDATION ENGINE (SHADOW OPTIMIZED)
# ==========================================
def validate_cone_color(roi_image):
    """
    Checks the bounding box provided by YOLO.
    Optimized to accept darker, shadowed cones by lowering Saturation and Value thresholds.
    """
    # Prevent crashing on empty boxes
    if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
        return False, 0.0

    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # Capture Red-Orange spectrum (Shadow Optimized)
    lower_red1 = np.array([0, 80, 40])
    upper_red1 = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)

    # Capture deep Red spectrum (Shadow Optimized)
    lower_red2 = np.array([160, 80, 40])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)

    # Combine masks
    final_mask = cv2.bitwise_or(mask1, mask2)

    # Calculate what percentage of the bounding box is red/orange
    red_pixel_count = cv2.countNonZero(final_mask)
    total_pixels = roi_image.shape[0] * roi_image.shape[1]

    color_percentage = (red_pixel_count / total_pixels) * 100

    # If at least 5% of the bounding box is red/orange, approve it
    is_valid = color_percentage > 5.0

    return is_valid, color_percentage


# ==========================================
# MAIN SYSTEM EXECUTION
# ==========================================
if __name__ == "__main__":

    print("\n[INFO] Initializing Hybrid AI + Color Mapping System...")
    transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3168", always_xy=True)
    transformer_to_gps = Transformer.from_crs("EPSG:3168", "EPSG:4326", always_xy=True)

    RAW_LON = 101.721667
    RAW_LAT = 3.008889
    CAM_X, CAM_Y = transformer_to_m.transform(RAW_LON, RAW_LAT)

    GROUND_ELEVATION = 50.0
    CAM_Z = GROUND_ELEVATION + 7.0

    CAM_AZIMUTH = 33.0
    CAM_TILT = 15.0

    IMG_WIDTH = 2688
    IMG_HEIGHT = 1520
    H_FOV = 84.0

    # --- LOAD YOLO AI ---
    print("[INFO] Loading Custom YOLOv8 Model...")
    model = YOLO("runs/detect/cone_detector-2/weights/best.pt")

    RTSP_URL = "rtsp://admin:uPM@2025@192.168.1.64:554/Streaming/Channels/101"

    print("[INFO] Attempting to connect to Hikvision Camera...")
    cap = cv2.VideoCapture(RTSP_URL)

    # --- PREVENT RTSP BUFFER OVERFLOW ---
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("[FAILED] ERROR: Could not connect to the camera.")
        exit()

    print("[SUCCESS] Connection Established! System is now live.")
    print("--------------------------------------------------\n")

    while True:
        ret, frame = cap.read()

        # --- AUTO-RECONNECT LOGIC ---
        if not ret:
            print("\n[WARNING] Video stream lagging or lost. Flushing buffer and reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            continue

        # --- RUN YOLO INFERENCE ---
        # Confidence dropped to 10% to catch the blurry distant cones!
        results = model(frame, verbose=False, conf=0.10, imgsz=1280)

        detected_centers = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                ai_confidence = float(box.conf[0])

                if cls_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    # --- THE HYBRID CHECK ---
                    # 1. Crop the image exactly where YOLO thinks the cone is
                    cropped_roi = frame[y1:y2, x1:x2]

                    # 2. Pass it to the Color Bouncer
                    color_approved, color_score = validate_cone_color(cropped_roi)

                    # 3. Only proceed if the Color Filter says YES
                    if color_approved:
                        u = int((x1 + x2) / 2)
                        v = int((y1 + y2) / 2)
                        detected_centers.append((u, v, ai_confidence, color_score))

                        # Draw Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Label showing both AI Confidence and Color Density
                        label = f"AI:{ai_confidence * 100:.0f}% | Color:{color_score:.0f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.circle(frame, (u, v), 5, (255, 0, 0), -1)

        # --- MAP THE CONFIRMED OBJECTS ---
        if detected_centers:
            for u, v, ai_conf, col_score in detected_centers:

                ray_vector = calculate_direction_vector(
                    u, v, IMG_WIDTH, IMG_HEIGHT, H_FOV, CAM_TILT, CAM_AZIMUTH
                )

                target_x, target_y = calculate_flat_ground_intersection(
                    CAM_X, CAM_Y, CAM_Z, ray_vector, GROUND_ELEVATION
                )

                if target_x and target_y:
                    target_lon, target_lat = transformer_to_gps.transform(target_x, target_y)

                    overlay_text = f"Lat: {target_lat:.5f}, Lon: {target_lon:.5f}"
                    cv2.putText(frame, overlay_text, (u - 100, v + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                else:
                    cv2.putText(frame, "Ray Above Horizon", (u - 50, v + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Hikvision Hybrid Mapping Engine", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n[INFO] Closing connection and shutting down.")
    cap.release()
    cv2.destroyAllWindows()