# tests/test_multidomain.py
import carla
import time
import os
import threading
from utils.lidar_utils import save_pointcloud
from config.generate_test_config import generate_target_vicinity_configs, generate_control_volume_configs

# === CONFIG ===
TARGET_BP_ID = 'static.prop.planea'
#TARGET_BP_ID = 'static.prop.planemesh01'
SAVE_DIR = './outputs/multidomain_test'
CAPTURE_FRAMES = 10
START_X = 10.0    # Start at 10m in front of LiDAR
END_X = 100.0     # Max test range
STEP_X = 10.0     # Step size along X-axis
Y_POS = 0.0       # Constant Y position
Z_POS = 1.7       # Prop height

# Define rotations (pitch, yaw, roll) and corresponding folder names
rotations = [
    ("rotation_1_pitch90_yaw0_roll0", carla.Rotation(pitch=90.0, yaw=0.0, roll=0.0)),
    ("rotation_2_pitch90_yaw60_roll0", carla.Rotation(pitch=90.0, yaw=60.0, roll=0.0)),
    ("rotation_3_pitch90_yaw0_roll60", carla.Rotation(pitch=60.0, yaw=0.0, roll=0.0)),
]

def run(client, world, lidar):
    print("[INFO] Starting Multi-Domain LiDAR Rotation-Distance Sweep Test")

    blueprint_library = world.get_blueprint_library()
    target_bp = blueprint_library.find(TARGET_BP_ID)
    lidar_location = lidar.get_transform().location

    for rot_name, rotation in rotations:
        print(f"[INFO] Processing {rot_name} with rotation {rotation}")

        rotation_dir = os.path.join(SAVE_DIR, rot_name)
        os.makedirs(rotation_dir, exist_ok=True)

        current_x = START_X

        while current_x <= END_X:
            distance_folder = f"{int(current_x)}m"
            dist_dir = os.path.join(rotation_dir, distance_folder)
            os.makedirs(dist_dir, exist_ok=True)

            # Synchronization primitives for capturing frames
            frame_lock = threading.Lock()
            frames_captured = 0
            frames_needed = CAPTURE_FRAMES
            capture_done = threading.Event()

            def lidar_callback(lidar_data):
                nonlocal frames_captured
                with frame_lock:
                    if frames_captured >= frames_needed:
                        return
                    save_path = os.path.join(dist_dir, f"frame_{frames_captured:03}.pcd")
                    save_pointcloud(lidar_data, save_path)
                    frames_captured += 1
                    print(f"[INFO] {rot_name} - {distance_folder}: Captured frame {frames_captured}/{frames_needed}")
                    if frames_captured >= frames_needed:
                        capture_done.set()

            lidar.listen(lidar_callback)

            # Spawn target with specific location and rotation
            new_location = carla.Location(x=lidar_location.x + current_x, y=Y_POS, z=Z_POS)
            prop_transform = carla.Transform(new_location, rotation)
            target = world.spawn_actor(target_bp, prop_transform)
            print(f"[INFO] Spawned target at {current_x:.1f}m with rotation {rotation}")

            # Wait for all frames to be captured
            while not capture_done.is_set():
                world.tick()
                time.sleep(0.1)

            target.destroy()
            print(f"[INFO] Removed target at {current_x:.1f}m")

            # Stop listening to avoid duplicate callbacks in next iteration
            lidar.stop()

            current_x += STEP_X

        # Generate configs after processing all distances for this rotation
        distances = list(range(int(START_X), int(END_X) + 1, int(STEP_X)))
        print(f"[INFO] Generating config files for {rot_name}")
        generate_target_vicinity_configs(rotation_dir, grade="A", distances=distances, rotation=(rotation.pitch, rotation.yaw, rotation.roll))

        generate_control_volume_configs(rotation_dir, distances=distances)

        print(f"[INFO] Completed rotation: {rot_name}")

    print("[SUCCESS] Multi-Domain LiDAR Rotation-Distance Sweep Test complete")
