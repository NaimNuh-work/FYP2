import numpy as np
import rasterio


def calculate_direction_vector(u, v, img_width, img_height, hov_deg, tilt_deg, azimuth_deg):
    """
    Translates a 2D pixel coordinate into a 3D normalized direction vector
    based on the camera's orientation.
    """
    # 1. Shift pixel origin (top-left) to image center (0,0)
    cx = img_width / 2.0
    cy = img_height / 2.0

    # Calculate focal length in pixels using the Horizontal Field of View (FOV)
    focal_length = cx / np.tan(np.radians(hov_deg) / 2.0)

    # Image coordinates: x is right, y is UP
    img_x = u - cx
    img_y = cy - v

    # 2. Base Vector (Camera looking North/Y-axis, perfectly level)
    # X = East/West, Y = North/South (Focal Length), Z = Up/Down
    vec = np.array([img_x, focal_length, img_y])
    vec = vec / np.linalg.norm(vec)  # Normalize the vector to length 1

    # 3. Apply Tilt (Pitch) - Rotate downward around the X-axis
    tilt = np.radians(tilt_deg)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(-tilt), -np.sin(-tilt)],
        [0, np.sin(-tilt), np.cos(-tilt)]
    ])

    # 4. Apply Azimuth (Yaw) - Rotate compass direction around the Z-axis
    azimuth = np.radians(-azimuth_deg)
    R_z = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])

    # Apply matrix rotations to the base vector
    vec_rotated = R_z @ R_x @ vec
    return vec_rotated


def perform_ray_trace(cam_x, cam_y, cam_z, direction_vector, dem_path, step_size=5.0, max_dist=10000.0):
    """
    Shoots the vector through the 3D space until it hits the DEM ground surface.
    Coordinates must be in a projected CRS (meters), like EPSG:3168.
    """
    try:
        with rasterio.open(dem_path) as dem:
            elevation_data = dem.read(1)  # Read the first band (altitude)
            transform = dem.transform  # The map coordinate transformer

            # Distance traveled along the ray
            t = 0.0

            while t < max_dist:
                # Calculate current 3D position of the ray
                current_x = cam_x + (t * direction_vector[0])
                current_y = cam_y + (t * direction_vector[1])
                current_z = cam_z + (t * direction_vector[2])

                # Convert real-world X,Y into DEM matrix rows and columns
                row, col = ~transform * (current_x, current_y)
                row, col = int(row), int(col)

                # Check if the ray has flown outside the DEM boundaries
                if row < 0 or row >= dem.height or col < 0 or col >= dem.width:
                    print("Ray exited the DEM area without hitting the ground.")
                    return None, None

                ground_z = elevation_data[row, col]

                # THE HIT CONDITION: If ray altitude is lower than ground altitude
                if current_z <= ground_z:
                    return current_x, current_y

                # Move forward by the step size (e.g., 2 to 5 meters)
                t += step_size

    except Exception as e:
        print(f"Error reading DEM: {e}")
        return None, None

    return None, None  # Failed to hit within max_dist


# ==========================================
# MAIN EXECUTION & 4-CAMERA CONFIGURATION
# ==========================================
if __name__ == "__main__":

    # 1. The Shared Pole Location (Stays the same for all cameras)
    POLE_X = 810000.0  # Example Easting in EPSG:3168
    POLE_Y = 350000.0  # Example Northing in EPSG:3168
    POLE_Z = 65.0  # Elevation + Pole Height

    # 2. The Shared Camera Hardware Specs
    IMG_WIDTH = 1920
    IMG_HEIGHT = 1080
    H_FOV = 90.0

    # 3. The 4-Camera Node Configuration Dictionary
    CAMERA_NODE_CONFIG = {
        "camera_north": {"azimuth": 0.0, "tilt": 15.0},
        "camera_east": {"azimuth": 90.0, "tilt": 15.0},
        "camera_south": {"azimuth": 180.0, "tilt": 15.0},
        "camera_west": {"azimuth": 270.0, "tilt": 15.0}
    }

    DEM_FILE = "peatland_dem_epsg3168.tif"  # Ensure this file is in your project folder

    # ------------------------------------------
    # Mock Input from YOLO Model
    # ------------------------------------------
    # In production, these variables will be passed directly from your YOLO script
    detected_camera_id = "camera_east"
    fire_pixel_u = 960
    fire_pixel_v = 800

    # ------------------------------------------
    # Execute the Math
    # ------------------------------------------
    print(f"--- Fire Detected on {detected_camera_id.upper()} ---")

    # Look up the correct angles for the specific camera that triggered the detection
    if detected_camera_id in CAMERA_NODE_CONFIG:
        active_azimuth = CAMERA_NODE_CONFIG[detected_camera_id]["azimuth"]
        active_tilt = CAMERA_NODE_CONFIG[detected_camera_id]["tilt"]

        print(f"Applying Calibration - Azimuth: {active_azimuth}°, Tilt: {active_tilt}°")

        # Calculate the 3D vector
        ray_vector = calculate_direction_vector(
            fire_pixel_u, fire_pixel_v,
            IMG_WIDTH, IMG_HEIGHT, H_FOV,
            active_tilt, active_azimuth
        )

        # Trace against the DEM
        print("Tracing ray against terrain model...")
        fire_x, fire_y = perform_ray_trace(
            POLE_X, POLE_Y, POLE_Z,
            ray_vector,
            DEM_FILE,
            step_size=2.0
        )

        if fire_x and fire_y:
            print(f"SUCCESS! Fire mapped at Coordinate -> X: {fire_x:.2f}, Y: {fire_y:.2f}")
            # NEXT STEP: Insert (fire_x, fire_y) into PostgreSQL/PostGIS database here
        else:
            print("Mapping failed. Fire coordinates could not be resolved on the ground.")

    else:
        print(f"Error: Camera ID '{detected_camera_id}' not found in configuration.")
        print(f"Error: Camera ID '{detected_camera_id}' not found in configuration.")