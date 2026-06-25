import cv2
import numpy as np
import os

# ==========================================
# RTSP OPTIMIZATION (Zero Delay)
# ==========================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def detect_cones(frame):
    """
    Detects orange cones in a frame using color thresholding and contour detection.
    Returns the frame with drawn boxes and a list of center coordinates (u, v).
    """
    # 1. Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define the color range for "Cone Orange" in HSV
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])

    # 3. Create a mask that isolates only the orange pixels
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Clean up the mask (remove small noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Find the outlines of the orange blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_centers = []

    for contour in contours:
        # 5. Filter by size. Lowered to 100 because your cone is far away.
        area = cv2.contourArea(contour)
        if area > 100:
            # Get the bounding box for the cone
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the exact center pixel (u, v)
            center_u = x + (w // 2)
            center_v = y + (h // 2)
            detected_centers.append((center_u, center_v))

            # Draw the box and center dot
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_u, center_v), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"({center_u}, {center_v})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_centers


# ==========================================
# CAMERA CONNECTION & EXECUTION
# ==========================================
if __name__ == "__main__":

    RTSP_URL = "rtsp://admin:uPM@2025@192.168.1.64:554/Streaming/Channels/101"

    print("\n==================================================")
    print("[INFO] Attempting to connect to Hikvision Camera...")
    print("[INFO] Please wait, this may take a few seconds...")
    print("==================================================\n")

    cap = cv2.VideoCapture(RTSP_URL)

    # TEST 1: Did the connection open at all?
    if not cap.isOpened():
        print("--------------------------------------------------")
        print("[FAILED] ERROR: Could not connect to the camera.")
        print("         Please check the following:")
        print("         1. Is the camera powered on?")
        print("         2. Is the ethernet cable connected securely?")
        print("         3. Is the IP address (192.168.1.64) correct?")
        print("         4. Are the username and password correct?")
        print("--------------------------------------------------")
        exit()

    # TEST 2: Can we actually read video data?
    ret, test_frame = cap.read()
    if not ret:
        print("--------------------------------------------------")
        print("[FAILED] ERROR: Connected to camera, but stream is empty.")
        print("         Check your RTSP channel or camera video settings.")
        print("--------------------------------------------------")
        cap.release()
        exit()

    # If both tests pass, it is 100% successful
    print("--------------------------------------------------")
    print("[SUCCESS] Connection Established!")
    print("[SUCCESS] Video stream is active and receiving frames.")
    print("--------------------------------------------------\n")

    print("Looking for cones... Press 'q' on the video window to quit.")

    while True:
        # Read the live frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("\n[WARNING] Video stream lost. Exiting...")
            break

        # Run the detection function
        processed_frame, cone_pixels = detect_cones(frame)

        # Display the output
        display_frame = cv2.resize(processed_frame, (1280, 720))
        cv2.imshow("Hikvision Cone Detection", display_frame)

        # Print the exact coordinates for your GIS mapping engine
        if cone_pixels:
            print(f"[DETECTED] Cone found at pixels: {cone_pixels}")

        # Press 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close
    print("\n[INFO] Closing connection and shutting down.")
    cap.release()
    cv2.destroyAllWindows()