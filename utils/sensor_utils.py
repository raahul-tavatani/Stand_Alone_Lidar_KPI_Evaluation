# utils/sensor_utils.py
import carla

def spawn_lidar(world, blueprint_library, location):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('horizontal_fov', '360')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1300000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov', '7')
    lidar_bp.set_attribute('lower_fov', '-16')
    
    transform = carla.Transform(location, carla.Rotation(pitch=0, yaw=0, roll=0))
    lidar = world.spawn_actor(lidar_bp, transform)

    if lidar is not None:
        print("[INFO] LiDAR sensor spawned successfully.")
    else:
        print("[ERROR] Failed to spawn LiDAR sensor.")

    return lidar
