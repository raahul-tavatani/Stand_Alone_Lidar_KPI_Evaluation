# utils/sensor_utils.py
import carla
import math

def spawn_lidar(world, blueprint_library, location,
                channels=64,
                rotation_frequency=10.0,
                points_per_second=7_300_000,      # try 2.6M or 5.0M
                horizontal_fov=120.0,             # concentrate rays; use 360.0 if you need full ring
                upper_fov=7.0,
                lower_fov=-16.0,
                range_m=200.0,
                sensor_tick=None):                 # default: 1/rotation_frequency
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    # Core scan parameters
    lidar_bp.set_attribute('channels', str(channels))
    lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))
    lidar_bp.set_attribute('points_per_second', str(points_per_second))
    lidar_bp.set_attribute('horizontal_fov', str(horizontal_fov))
    lidar_bp.set_attribute('upper_fov', str(upper_fov))
    lidar_bp.set_attribute('lower_fov', str(lower_fov))
    lidar_bp.set_attribute('range', str(range_m))

    # Align sensor tick to one full revolution by default
    if sensor_tick is None:
        sensor_tick = 1 / rotation_frequency
    lidar_bp.set_attribute('sensor_tick', str(sensor_tick))

    # Make returns as inclusive as possible (avoid CARLA’s intensity drop-off)
    #lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    #lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    #lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    #lidar_bp.set_attribute('noise_stddev', '0.0')  # set >0 if you want noise

    transform = carla.Transform(location, carla.Rotation(pitch=0, yaw=0, roll=0))
    lidar = world.spawn_actor(lidar_bp, transform)

    # Sanity: expected rays per frame
    rays_per_frame = int(points_per_second / rotation_frequency)
    per_channel = rays_per_frame / channels
    horiz_step_deg = (horizontal_fov / per_channel) if per_channel > 0 else float('nan')

    print(f"[INFO] LiDAR spawned.")
    print(f"       Expected rays/frame: {rays_per_frame:,}")
    print(f"       Per channel samples: {per_channel:.1f}")
    print(f"       Horizontal step ≈ {horiz_step_deg:.3f}°")

    return lidar
