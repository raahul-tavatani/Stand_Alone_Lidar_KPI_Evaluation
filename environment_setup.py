import carla
import time
import subprocess
import os
import open3d as o3d
import numpy as np

CARLA_PATH = r"C:\carla\CARLA_Updated\WindowsNoEditor\CarlaUE4.exe"

def launch_carla(port):
    return subprocess.Popen([
        CARLA_PATH,
        f"-carla-rpc-port={port}",
        "-windowed", "-ResX=800", "-ResY=600"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    print("🚗 Launching CARLA on port 2000...")
    carla_process = launch_carla(2000)
    print("⏳ Waiting for CARLA to start...")
    time.sleep(8)

    client = carla.Client('localhost', 2000)
    client.set_timeout(25.0)

    print("🌍 Loading map: Lidar_Testing_Ground...")
    world = client.load_world('Lidar_Testing_Ground')

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.010
    settings.synchronous_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Define LiDAR spawn location
    lidar_location = carla.Location(x=0.0, y=0.0, z=1.7)
    lidar_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)

    # LiDAR blueprint and settings
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    # Geometry/scan budget
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '7300000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('sensor_tick', '0.1')

    # Coverage
    lidar_bp.set_attribute('horizontal_fov', '360')
    lidar_bp.set_attribute('upper_fov', '7')    # total 23°
    lidar_bp.set_attribute('lower_fov', '-16')
    lidar_bp.set_attribute('range', '100')

    # IMPORTANT: eliminate artificial point drop/attenuation
    lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.0')
    lidar_bp.set_attribute('noise_stddev', '0.0')  # optional: 0 for determinism


    # Spawn LiDAR
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar_actor = world.spawn_actor(lidar_bp, lidar_transform)
    print("✅ LiDAR spawned at", lidar_location)

    # Spawn plane 30 meters in front
    plane_bp = blueprint_library.find('static.prop.planea')
    if not plane_bp:
        print("❌ Plane blueprint 'static.prop.planea' not found!")
        lidar_actor.destroy()
        carla_process.kill()
        return

    plane_location = lidar_location + carla.Location(x=30.0, y=0.0, z=0.0)
    plane_transform = carla.Transform(location=plane_location)
    plane_actor = world.spawn_actor(plane_bp, plane_transform)
    print(f"✅ Plane spawned at {plane_location}")

    frame_received = False

    def lidar_callback(data):
        nonlocal frame_received
        if frame_received:
            return

        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        filename = "lidar_frame_0000.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"📁 Saved LiDAR point cloud to '{filename}' with {len(points)} points.")

        frame_received = True
        # Clean up
        lidar_actor.stop()
        lidar_actor.destroy()
        plane_actor.destroy()
        carla_process.kill()
        print("🧹 Cleaned up and exiting.")

    lidar_actor.listen(lidar_callback)

    print("📡 Waiting for LiDAR data...")
    try:
        while not frame_received:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("⚠️ Interrupted. Cleaning up...")
        lidar_actor.destroy()
        plane_actor.destroy()
        carla_process.kill()

if __name__ == "__main__":
    main()
